# models/conv3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return F.relu(out)

class Conv3D(nn.Module):
    def __init__(self):
        super(Conv3D, self).__init__()
        self.res_blocks = nn.Sequential(
            ResidualBlock3D(3, 192),  # Change input channels from 96 to 3
            nn.MaxPool3d(2),
            ResidualBlock3D(192, 384),
            nn.MaxPool3d(2),
            ResidualBlock3D(384, 512),
            nn.MaxPool3d(2),
            ResidualBlock3D(512, 384),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ResidualBlock3D(384, 192),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ResidualBlock3D(192, 96),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ResidualBlock3D(96, 48)
        )

    def forward(self, x):
        x = self.res_blocks(x)
        return x

if __name__ == "__main__":
    model = Conv3D()
    print(model)
