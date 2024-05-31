# models/appearance_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(out_channels // 8, 32), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(out_channels // 8, 32), num_channels=out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return F.relu(out)

class AppearanceEncoder(nn.Module):
    def __init__(self):
        super(AppearanceEncoder, self).__init__()
        self.initial = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.res_blocks = nn.Sequential(
            ResidualBlock2D(64, 128),
            nn.AvgPool2d(2),
            ResidualBlock2D(128, 256),
            nn.AvgPool2d(2),
            ResidualBlock2D(256, 512),
            nn.AvgPool2d(2),
            ResidualBlock2D(512, 1024)  # Adjusted to ensure dimensions are compatible
        )
        self.reshaped = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups=min(1024 // 8, 32), num_channels=1024)
        )
        self.res3d_blocks = nn.Sequential(
            ResidualBlock2D(1024, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96)
        )

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.res_blocks(x)
        x = F.relu(self.reshaped(x))
        x = x.view(x.size(0), 96, 16, 16, 16)  # Reshape to 3D features
        x = self.res3d_blocks(x)
        return x

if __name__ == "__main__":
    model = AppearanceEncoder()
    print(model)
