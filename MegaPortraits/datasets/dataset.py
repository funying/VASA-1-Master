# datasets/dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2

class MegaPortraitDataset(Dataset):
    def __init__(self, data_path, transform=None, max_frames=60):
        self.data_path = data_path
        self.transform = transform
        self.data_info = self.load_data_info()
        self.same_person_pairs = [pair for pair in self.data_info if pair['same_person']]
        self.different_person_pairs = [pair for pair in self.data_info if not pair['same_person']]
        self.max_frames = max_frames  # Maximum number of frames to pad/truncate to

    def load_data_info(self):
        with open(os.path.join(self.data_path, 'driving_video.json'), 'r') as f:
            data_info = json.load(f)
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if idx < len(self.same_person_pairs):
            sample_info = self.same_person_pairs[idx]
        else:
            sample_info = self.different_person_pairs[idx - len(self.same_person_pairs)]

        source_path = os.path.join(self.data_path, sample_info['source'])
        driving_path = os.path.join(self.data_path, sample_info['driving'])

        source_image = Image.open(source_path).convert('RGB')
        driving_frames = self.load_video(driving_path)

        if self.transform:
            source_image = self.transform(source_image)
            driving_frames = [self.transform(frame) for frame in driving_frames]

        # Pad or truncate driving frames to max_frames
        if len(driving_frames) < self.max_frames:
            pad_size = self.max_frames - len(driving_frames)
            padding = [torch.zeros_like(driving_frames[0]) for _ in range(pad_size)]
            driving_frames.extend(padding)
        else:
            driving_frames = driving_frames[:self.max_frames]

        return source_image, torch.stack(driving_frames)

    def load_video(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).resize((224, 224))  # Ensure all frames are 224x224
            frames.append(frame)
        video_capture.release()
        return frames

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = MegaPortraitDataset(data_path='/data', transform=transform)
    print(f'Dataset size: {len(dataset)}')
    sample = dataset[0]
    print(f'Sample shapes: {sample[0].shape}, {sample[1].shape}')
