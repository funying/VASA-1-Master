# models/student_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

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

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class StudentEncoder(nn.Module):
    def __init__(self):
        super(StudentEncoder, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final classification layer

    def forward(self, x):
        return self.resnet(x)

class StudentGenerator(nn.Module):
    def __init__(self):
        super(StudentGenerator, self).__init__()
        self.res_blocks = nn.ModuleList([
            self._make_layer(192, 192, 3),
            self._make_layer(192, 192, 3),
            self._make_layer(192, 192, 3),
            self._make_layer(192, 96, 3),
            self._make_layer(96, 48, 3),
            self._make_layer(48, 24, 3)
        ])
        self.final = nn.Sequential(
            nn.InstanceNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 3, kernel_size=1)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock2D(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, segmap):
        for layer in self.res_blocks:
            x = layer(x)
        x = self.final(x)
        return torch.sigmoid(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.encoder = StudentEncoder()
        self.generator = StudentGenerator()

    def forward(self, x, segmap):
        x = self.encoder(x)
        x = self.generator(x, segmap)
        return x

if __name__ == "__main__":
    model = StudentModel()
    print(model)
