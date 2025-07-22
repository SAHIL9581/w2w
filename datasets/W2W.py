import json
import os
from os.path import join as pjoin
import numpy as np
from torch.utils import data
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


class W2WDataset(data.Dataset):
    def __init__(self, args, split="train"):
        self.root = args.root
        self.patch_height = args.patch_height
        self.split = split
        self.image_list = os.listdir(pjoin(self.root, self.split, "X"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        name = self.image_list[index][:-4]
        inp_path = pjoin(self.root, self.split, "X", name + ".npy")
        gt_path = pjoin(self.root, self.split, "Y", name + ".json")

        inp_image = np.load(inp_path, allow_pickle = True).astype(np.float32)
        inp_image = np.expand_dims(inp_image, 0)

        f = open(gt_path)
        data = json.load(f)

        label = {}
        lbl = []
        top = []
        height = []
    
        for i in data:
            top.append(data[i]["Top"] / self.patch_height)  # normalizing dip between 0-1
            height.append(data[i]["Height"] / self.patch_height)  # normalizing azimuth between 0-1 
            lbl.append(1)

        label["lbl"] = lbl
        label["Top"] = top
        label["Height"] = height

        target = {}
        target["labels"] = torch.tensor(label["lbl"])
        t, h = (
            torch.tensor(label["Top"]),
            torch.tensor(label["Height"]),
        )
        t, h = t.view(-1, 1), h.view(-1, 1)
        target["loc_info"] = torch.hstack((t, h))

        return inp_image, target


class W2WRandWellFromRawData(data.Dataset):
    def __init__(self, args, seed = None):
        self.raw_data_path = args.raw_data_path
        self.label_encoder_path = args.label_encoder_path
        self.columns_in_data = ['WELL', 'DEPTH_MD', 'X_LOC', 'Y_LOC', 'GROUP', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP']
        self.patch_height = args.patch_height
        self.seed = seed if seed else np.random.randint(2**32 - 1)
        # print("Seed being used {}".format(self.seed))

        x, rand_well_label = self.getXyFromRandWell()
        self.inp_image = x
        self.gt = rand_well_label

    def load_data(self, path, delimiter = ';'):
        return pd.read_csv(path, delimiter = delimiter)

    def count_well(self, data):
        return data.WELL.value_counts().shape[0]

    def get_well_names(self, data):
        return list(data.WELL.value_counts().index)

    def get_random_well(self, data, seed):

        np.random.seed(seed)
        rand_well_index = np.random.randint(0, self.count_well(data))
        rand_well_name = self.get_well_names(data)[rand_well_index]
        print('Displaying information for Well {}'.format(rand_well_name))
        rand_well_data = data[data['WELL'] == rand_well_name]

        return rand_well_data

    def remove_force_column(self, data):
        data.drop(['FORCE_2020_LITHOFACIES_LITHOLOGY', 'FORCE_2020_LITHOFACIES_CONFIDENCE'], axis = 1, inplace = True)

    def remove_zloc_column(self, data):
        data.drop(['Z_LOC'], axis = 1, inplace = True)

    def remove_formation_column(self, data):
        data.drop(['FORMATION'], axis = 1, inplace = True)

    def fill_group_na_value(self, data, well_name, method = 'bfill'):
        return data[data['WELL'] == well_name].GROUP.fillna(method = method)

    def fill_xy_loc(self, data, well_names, method = 'bfill'):

        for well in well_names:
            if self.percet_missing_data(data[data.WELL == well].X_LOC) != 0:
                data.X_LOC.loc[data[data['WELL'] == well].X_LOC.index] = data[data['WELL'] == well].X_LOC.fillna(method=method)
                data.Y_LOC.loc[data[data['WELL'] == well].Y_LOC.index] = data[data['WELL'] == well].Y_LOC.fillna(method=method)

    def percet_missing_data(self, dataframe):
        return dataframe.isna().sum()/dataframe.shape[0]*100

    def get_labels(self, data):
        return data.GROUP

    def drop_labels(self, data):
        return data.drop(['GROUP'], axis = 1, inplace = True)

    def fill_na(self, data, value = 0):
        return data.fillna(value, inplace = True)

    def get_gt(self, Y):
        gts = []
        for sample_num, y in enumerate(Y):
            gt = {}
            count = 0
            kink = [i+1 for i in range(len(y)-1) if not y[i] == y[i+1]]
            kink.insert(0, 0)
            gp = [y[idx] for idx in kink]
            top = kink.copy()
            kink.append(len(y))
            height = [element1 - element2 for (element1, element2) in zip(kink[1:], kink[:-1])]
            for t, h, g in zip(top, height, gp):
                temp = {}
                temp['Group'] = int(g)
                temp['Top'] = int(t)
                temp['Height'] = int(h)
                gt[count] = temp
                count+=1

            gts.append(gt)
            
        return gts

    def getXyFromRandWell(self):

        well_data = self.load_data(self.raw_data_path)
        rand_well = self.get_random_well(well_data, self.seed)
        label_encoder_dict = json.load(open(self.label_encoder_path))

        well_name = rand_well.WELL.iloc[0]
        self.remove_force_column(rand_well)
        self.remove_zloc_column(rand_well)
        self.remove_formation_column(rand_well)
        rand_well = rand_well[self.columns_in_data]
        rand_well.GROUP.loc[rand_well[rand_well['WELL'] == rand_well.WELL.iloc[0][0]].GROUP.index] = self.fill_group_na_value(rand_well, well_name, method = 'bfill')
        rand_well.GROUP.loc[rand_well[rand_well['WELL'] == rand_well.WELL.iloc[0][0]].GROUP.index] = self.fill_group_na_value(rand_well, well_name, method = 'ffill')
        self.fill_xy_loc(rand_well, well_name, method = 'bfill')
        self.fill_xy_loc(rand_well, well_name, method = 'ffill')
        rand_well_label = self.get_labels(rand_well) 
        self.drop_labels(rand_well)
        self.fill_na(rand_well)

        rand_well_label.replace(label_encoder_dict, inplace = True)
        rand_well.drop(['WELL'], axis = 1, inplace = True)

        scaler = StandardScaler()
        scaler.fit(rand_well)
        rand_well = scaler.transform(rand_well)

        idx = list(range(0, rand_well.shape[0], self.patch_height))
        x = np.asarray([rand_well[i:i+self.patch_height] for i in idx if rand_well[i:i+self.patch_height].shape[0] == self.patch_height]).astype(np.float32)
        y = np.asarray([rand_well_label.values[i:i+self.patch_height] for i in idx if rand_well_label.values[i:i+self.patch_height].shape[0] == self.patch_height])

        y = self.get_gt(y)

        return x, y

    def __len__(self):
        return len(self.inp_image)

    def __getitem__(self, index):

        inp_image = np.expand_dims(self.inp_image[index], 0)
        data = self.gt[index]

        label = {}
        lbl = []
        top = []
        height = []
    
        for i in data:
            
            top.append(data[i]["Top"] / self.patch_height)  # normalizing dip between 0-1
            height.append(data[i]["Height"] / self.patch_height)  # normalizing azimuth between 0-1 
            lbl.append(1)

        label["lbl"] = lbl
        label["Top"] = top
        label["Height"] = height

        target = {}
        target["labels"] = torch.tensor(label["lbl"])
        t, h = (
            torch.tensor(label["Top"]),
            torch.tensor(label["Height"]),
        )
        t, h = t.view(-1, 1), h.view(-1, 1)
        target["loc_info"] = torch.hstack((t, h))

        return inp_image, target