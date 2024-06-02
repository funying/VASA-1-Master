# losses/perceptual_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import tensorflow as tf

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vggface = VGGFace(include_top=False, input_shape=(224, 224, 3))
        self.model = Model(inputs=self.vggface.input, outputs=self.vggface.get_layer('conv5_3').output)
        for param in self.model.layers:
            param.trainable = False

    def forward(self, x, y):
        # Convert torch tensor to numpy and resize
        x = x.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)  # Convert to (batch_size, height, width, channels) and float32
        y = y.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)  # Convert to (batch_size, height, width, channels) and float32

        # Debugging: print the shapes and types of x and y
        print(f"x shape before resizing: {x.shape}, dtype: {x.dtype}")
        print(f"y shape before resizing: {y.shape}, dtype: {y.dtype}")

        try:
            # Ensure both tensors have the same shape by resizing both to (224, 224)
            x_resized = np.array([cv2.resize(frame, (224, 224)) for frame in x])
            y_resized = np.array([cv2.resize(frame, (224, 224)) for frame in y])
        except cv2.error as e:
            print(f"OpenCV error during resizing: {e}")
            print(f"x shape: {x.shape}, y shape: {y.shape}")
            raise

        # Debugging: print the shapes and types of resized x and y
        print(f"x_resized shape after resizing: {x_resized.shape}, dtype: {x_resized.dtype}")
        print(f"y_resized shape after resizing: {y_resized.shape}, dtype: {y_resized.dtype}")

        # Ensure the resized arrays have 3 channels
        if x_resized.shape[-1] != 3:
            x_resized = np.stack([x_resized[..., i] for i in range(3)], axis=-1)
        if y_resized.shape[-1] != 3:
            y_resized = np.stack([y_resized[..., i] for i in range(3)], axis=-1)

        # Preprocess input
        x_resized = preprocess_input(x_resized)
        y_resized = preprocess_input(y_resized)

        # Convert numpy array back to torch tensor
        x_resized = torch.tensor(x_resized.copy(), dtype=torch.float32).permute(0, 3, 1, 2)
        y_resized = torch.tensor(y_resized.copy(), dtype=torch.float32).permute(0, 3, 1, 2)

        # Ensure correct shape for Keras model
        x_resized = x_resized.permute(0, 2, 3, 1).numpy()  # Convert to (batch_size, height, width, channels)
        y_resized = y_resized.permute(0, 2, 3, 1).numpy()  # Convert to (batch_size, height, width, channels)

        # Debugging: print the shapes and types of resized x and y before passing to the model
        print(f"x_resized shape before model: {x_resized.shape}, dtype: {x_resized.dtype}")
        print(f"y_resized shape before model: {y_resized.shape}, dtype: {y_resized.dtype}")

        # Forward through the model
        x_vgg = self.model.predict(x_resized)
        y_vgg = self.model.predict(y_resized)
        x_vgg = torch.tensor(x_vgg)
        y_vgg = torch.tensor(y_vgg)

        return F.l1_loss(x_vgg, y_vgg)
        
if __name__ == "__main__":
    loss = PerceptualLoss()
    print(loss)
