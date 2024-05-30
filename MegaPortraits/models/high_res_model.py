# models/high_res_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
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

class HighResEncoder(nn.Module):
    def __init__(self):
        super(HighResEncoder, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.res_blocks = nn.Sequential(
            ResidualBlock2D(64, 128),
            nn.AvgPool2d(2),
            ResidualBlock2D(128, 256)
        )
        self.conv_final = nn.Conv2d(256, 1536, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.res_blocks(x)
        x = self.conv_final(x)
        return x.view(x.size(0), 1536, 16, 16, 16)  # Reshape to 3D features

class HighResDecoder(nn.Module):
    def __init__(self):
        super(HighResDecoder, self).__init__()
        self.res_blocks = nn.Sequential(
            ResidualBlock2D(1536, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(32, 16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock2D(16, 3)
        )

    def forward(self, x):
        x = self.res_blocks(x)
        return torch.sigmoid(x)

class HighResModel(nn.Module):
    def __init__(self):
        super(HighResModel, self).__init__()
        self.encoder = HighResEncoder()
        self.decoder = HighResDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    model = HighResModel()
    print(model)
