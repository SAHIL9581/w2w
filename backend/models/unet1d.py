import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    """Returns the activation function based on the provided name."""
    return nn.PReLU() if name == 'prelu' else nn.ReLU() if name == 'relu' else nn.GELU()

class Block1D(nn.Module):
    """A basic 1D convolutional block with two convolution layers."""
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3, activation='prelu'):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels), get_activation(activation),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels), get_activation(activation)
        )
    def forward(self, x): return self.b(x)

class UNet1D(nn.Module):
    """A 1D U-Net architecture with robust skip connections."""
    def __init__(self, in_channels, activation='prelu'):
        super().__init__()
        self.start = Block1D(in_channels, 32, stride=1, activation=activation)
        self.e1 = Block1D(32, 64, stride=2, activation=activation)
        self.e2 = Block1D(64, 128, stride=2, activation=activation)
        self.e3 = Block1D(128, 256, stride=2, activation=activation)
        self.mid = Block1D(256, 512, stride=2, activation=activation)
        self.uc3 = nn.ConvTranspose1d(512, 256, 2, 2)
        self.d3 = Block1D(512, 256, stride=1, activation=activation)
        self.uc2 = nn.ConvTranspose1d(256, 128, 2, 2)
        self.d2 = Block1D(256, 128, stride=1, activation=activation)
        self.uc1 = nn.ConvTranspose1d(128, 64, 2, 2)
        self.d1 = Block1D(128, 64, stride=1, activation=activation)
        self.uc0 = nn.ConvTranspose1d(64, 32, 2, 2)
        self.d0 = Block1D(64, 32, stride=1, activation=activation)
        self.out_conv = nn.Conv1d(32, in_channels, 1)

    def forward(self, x):
        s1 = self.start(x); s2 = self.e1(s1); s3 = self.e2(s2); s4 = self.e3(s3); m = self.mid(s4)
        d3 = self.d3(torch.cat((F.interpolate(self.uc3(m), size=s4.shape[2]), s4), 1))
        d2 = self.d2(torch.cat((F.interpolate(self.uc2(d3), size=s3.shape[2]), s3), 1))
        d1 = self.d1(torch.cat((F.interpolate(self.uc1(d2), size=s2.shape[2]), s2), 1))
        d0 = self.d0(torch.cat((F.interpolate(self.uc0(d1), size=s1.shape[2]), s1), 1))
        return self.out_conv(d0)

class UNetEncoder1D(nn.Module):
    """The encoder part of the 1D U-Net."""
    def __init__(self, in_channels, activation='prelu'):
        super().__init__()
        self.start = Block1D(in_channels, 32, stride=1, activation=activation)
        self.e1 = Block1D(32, 64, stride=2, activation=activation)
        self.e2 = Block1D(64, 128, stride=2, activation=activation)
        self.e3 = Block1D(128, 256, stride=2, activation=activation)
        self.mid = Block1D(256, 512, stride=2, activation=activation)
    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1); s1 = self.start(x); s2 = self.e1(s1); s3 = self.e2(s2); s4 = self.e3(s3); return self.mid(s4)
