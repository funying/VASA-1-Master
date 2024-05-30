# utils/visualization.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def save_image(tensor, path):
    """Saves a tensor as an image file."""
    image = transforms.ToPILImage()(tensor.cpu().detach())
    image.save(path)

def show_image(tensor, title=None):
    """Displays a tensor as an image."""
    image = transforms.ToPILImage()(tensor.cpu().detach())
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    dummy_image = torch.rand(3, 224, 224)
    save_image(dummy_image, 'dummy_image.png')
    show_image(dummy_image, 'Dummy Image')
