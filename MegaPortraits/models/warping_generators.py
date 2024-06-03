# models/warping_generators.py

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

class WarpingGenerator(nn.Module):
    def __init__(self):
        super(WarpingGenerator, self).__init__()
        self.initial = nn.Conv3d(152, 2048, kernel_size=1, stride=1) 
        self.res_blocks = nn.Sequential(
            ResidualBlock3D(2048, 256),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(256, 128),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(128, 64),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(64, 32),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(32, 16),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        )
        self.final = nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, R, t, z, e):
        print(f"R shape: {R.shape}, t shape: {t.shape}, z shape: {z.shape}, e shape: {e.shape}")
        x = torch.cat((R, t, z, e), dim=1)
        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # singleton dimensions for depth, height, width
        print(f"Input shape after concatenation and reshaping: {x.shape}")
        x = F.relu(self.initial(x))
        x = self.res_blocks(x)
        x = self.final(x)
        return self.tanh(x)

if __name__ == "__main__":
    model = WarpingGenerator()
    print(model)
