# losses/perceptual_loss.py

import torch
import torch.nn as nn
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vggface = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3)).features
        for param in self.vggface.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vggface(x)
        y_vgg = self.vggface(y)
        loss_vgg = self.l1_loss(x_vgg, y_vgg)
        return loss_vgg

if __name__ == "__main__":
    loss = PerceptualLoss()
    print(loss)
