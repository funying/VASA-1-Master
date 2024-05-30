# datasets/augmentations.py

from torchvision import transforms

def get_augmentations():
    return transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
