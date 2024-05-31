# losses/perceptual_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.models import Model

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vggface = VGGFace(include_top=False, input_shape=(224, 224, 3))
        self.model = Model(inputs=self.vggface.input, outputs=self.vggface.get_layer('conv5_3').output)
        for param in self.model.layers:
            param.trainable = False

    def forward(self, x, y):
        x_vgg = self.model(preprocess_input(x))
        y_vgg = self.model(preprocess_input(y))
        return F.l1_loss(x_vgg, y_vgg)


if __name__ == "__main__":
    loss = PerceptualLoss()
    print(loss)
