# models/discriminator.py

import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (input_channels) x 256 x 256
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 31 x 31
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            # state size. 1 x 30 x 30
        )

    def forward(self, input):
        return self.model(input)
