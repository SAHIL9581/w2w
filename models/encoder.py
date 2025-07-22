import torch.nn as nn
from models.utils import get_activation


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_name, kernel_size=3):
        super(Block, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            get_activation(act_name),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            get_activation(act_name),
        )

    def forward(self, x):
        residual = self.block(x)
        return residual


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, act_name):
        super(UNetEncoder, self).__init__()

        self.num_channels = in_channels
        self.start = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            get_activation(act_name)
        )

        self.encode1 = Block(32, 64, 1, act_name)
        self.encode2 = Block(64, 128, 2, act_name)
        self.encode3 = Block(128, 256, 2, act_name)

        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride = 2),
            nn.BatchNorm2d(512),
            get_activation(act_name)
        )

    def forward(self, x):
        x = self.start(x)

        # Encoder
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)

        out_middle = self.middle(x3)

        return out_middle
        # return out_middle