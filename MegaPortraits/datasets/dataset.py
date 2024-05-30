# datasets/dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

class MegaPortraitDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data_info = self.load_data_info()

    def load_data_info(self):
        with open(os.path.join(self.data_path, 'driving_video.json'), 'r') as f:
            data_info = json.load(f)
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        sample_info = self.data_info[idx]
        source_path = os.path.join(self.data_path, sample_info['source'])
        driving_path = os.path.join(self.data_path, sample_info['driving'])

        source_image = Image.open(source_path).convert('RGB')
        driving_image = Image.open(driving_path).convert('RGB')

        if self.transform:
            source_image = self.transform(source_image)
            driving_image = self.transform(driving_image)

        return source_image, driving_image

    def center_crop(self, image):
        width, height = image.size
        new_width, new_height = 224, 224
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        image = image.crop((left, top, right, bottom))
        return image

    def random_warp(self, image):
        image_np = np.array(image)
        src_points = np.float32([[50, 50], [200, 50], [50, 200]])
        dst_points = src_points + np.random.uniform(-10, 10, src_points.shape).astype(np.float32)
        tform = cv2.getAffineTransform(src_points, dst_points)
        warped = cv2.warpAffine(image_np, tform, (image_np.shape[1], image_np.shape[0]))
        return Image.fromarray(warped)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = MegaPortraitDataset(data_path='/path/to/data', transform=transform)
    print(f'Dataset size: {len(dataset)}')
    sample = dataset[0]
    print(f'Sample shapes: {sample[0].shape}, {sample[1].shape}')
