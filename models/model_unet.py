import torch.nn as nn
import torch
from models.utils import  get_activation

class Block(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, stride = 2, kernel_size=3, activation = 'prelu'):
        super(Block, self).__init__()
        padding = kernel_size // 2
        self.act =  get_activation(activation)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            self.act,
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            self.act,
        )

    def forward(self, x):
        residual = self.block(x)
        return residual


class UNet(nn.Module):
    def __init__(self, in_channels=3, activation = 'prelu'):
        super(UNet, self).__init__()

        self.act =  get_activation(activation)

        self.start = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            self.act
        )

        self.encode1 = Block(32, 64, 1, activation=activation)
        self.encode2 = Block(64, 128, 2, activation=activation)
        self.encode3 = Block(128, 256, 2, activation=activation)

        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride = 2),
            nn.BatchNorm2d(512),
            self.act
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1)
        self.dencode3 = Block(512, 256, 1, activation=activation)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)
        self.dencode2 = Block(256, 128, 1, activation=activation)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.dencode1 = Block(128, 64, 1, activation=activation)

        self.end = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=in_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.start(x)

        # Encoder
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)

        out_middle = self.middle(x3)

        # Decoder
        out3 = self.upconv3(out_middle, output_size = x3.size())
        out3 = torch.cat((out3, x3), dim=1)
        out3 = self.dencode3(out3)

        out2 = self.upconv2(out3, output_size = x2.size())
        out2 = torch.cat((out2, x2), dim=1)
        out2 = self.dencode2(out2)

        out1 = self.upconv1(out2, output_size = x1.size())
        out1 = torch.cat((out1, x1), dim=1)
        out1 = self.dencode1(out1)

        out = self.end(out1)
        out = self.out(out)

        return out