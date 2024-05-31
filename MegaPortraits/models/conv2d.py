# models/conv2d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(ResidualBlock2D, self).__init__()
        if out_channels % num_groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by num_groups ({num_groups})')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
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

class Conv2D(nn.Module):
    def __init__(self):
        super(Conv2D, self).__init__()
        self.reshaped = nn.Conv2d(96 * 16, 1536, kernel_size=1, stride=1)
        self.res_blocks = nn.Sequential(
            ResidualBlock2D(1536, 512, num_groups=8),  # 512 is divisible by 8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(512, 256, num_groups=8),  # 256 is divisible by 8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(256, 128, num_groups=8),  # 128 is divisible by 8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(128, 64, num_groups=8),   # 64 is divisible by 8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(64, 32, num_groups=8)    # 32 is divisible by 8
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))  # Reshape to 2D features
        x = F.relu(self.reshaped(x))
        x = self.res_blocks(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    model = Conv2D()
    print(model)

