from torch import nn
import numpy as np
from models.encoder import UNetEncoder
from models.transformer import Project, Query, Transformer
from models.utils import get_activation

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # self.project_in_features = int(np.ceil(args.patch_height/2/2/2)) * int(np.ceil(args.num_input_logs/2/2/2))
        self.encoder = UNetEncoder(in_channels = args.in_channels, act_name = args.act_name)
        self.project = Project(in_features = args.project_in_features, out_features = args.hidden_dim)
        self.create_query = Query(seq_len = args.num_queries, dim = args.hidden_dim)
        self.transformer_module = Transformer(in_features = args.hidden_dim, num_heads = args.num_heads, dropout = args.dropout, expansion_factor = args.expansion_factor, act_name = args.act_name)
        self.transformer_list = [self.transformer_module for _ in range(args.num_transformers)]
        self.transformers = nn.ModuleList(self.transformer_list)
        self.finalize = nn.Sequential(nn.Linear(args.hidden_dim, args.output_size), get_activation(args.act_name), nn.LayerNorm(args.output_size))

    def forward(self, img):
        img = self.encoder(img)
        seq = self.project(img)
        learned_query = self.create_query(seq)
        for transformers in self.transformers:
            seq = transformers(learned_query, seq)
        return self.finalize(seq)

def build_model(args):
    return Model(args)