# scripts/train_highres_model.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.high_res_model import HighResModel
from models.discriminator import PatchGANDiscriminator
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial_loss import AdversarialLoss
from losses.cycle_consistency_loss import CycleConsistencyLoss
from utils.logger import setup_logger
from datasets.dataset import MegaPortraitDataset

class HighResModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = HighResModel().to(self.device)
        self.discriminator = PatchGANDiscriminator().to(self.device)

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.adversarial_loss = AdversarialLoss().to(self.device)
        self.cycle_consistency_loss = CycleConsistencyLoss().to(self.device)

        self.optimizer_G = optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            betas=(0.5, 0.999),
            eps=1e-8,
            weight_decay=1e-2
        )

        self.optimizer_D = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config['lr'],
            betas=(0.5, 0.999),
            eps=1e-8,
            weight_decay=1e-2
        )

    def load_data(self):
        transform = transforms.Compose([
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        train_dataset = MegaPortraitDataset(self.config['data_path'], transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)

    def train(self):
        for epoch in range(self.config['epochs']):
            for i, data in enumerate(self.train_loader):
                source, target = data
                source = source.to(self.device)
                target = target.to(self.device)

                # Determine if we should use same person pairs or different person pairs
                if i % 2 == 0:
                    # Use pairs from same_person_pairs for loss functions that require same person
                    sample_info = self.train_loader.dataset.same_person_pairs[i % len(self.train_loader.dataset.same_person_pairs)]
                else:
                    # Use pairs from different_person_pairs for other loss functions
                    sample_info = self.train_loader.dataset.different_person_pairs[i % len(self.train_loader.dataset.different_person_pairs)]

                source_path = os.path.join(self.train_loader.dataset.data_path, sample_info['source'])
                driving_path = os.path.join(self.train_loader.dataset.data_path, sample_info['driving'])

                source_image = Image.open(source_path).convert('RGB')
                driving_frames = self.train_loader.dataset.load_video(driving_path)

                if self.train_loader.dataset.transform:
                    source_image = self.train_loader.dataset.transform(source_image)
                    driving_frames = [self.train_loader.dataset.transform(frame) for frame in driving_frames]

                source = source_image.to(self.device)
                target = driving_frames.to(self.device)

                # Generator forward pass
                self.optimizer_G.zero_grad()

                output = self.model(source)

                # Calculate losses
                loss_l1 = self.l1_loss(output, target)
                loss_adv = self.adversarial_loss(self.discriminator(target), self.discriminator(output))
                loss_perceptual = self.perceptual_loss(output, target)
                loss_cycle = self.cycle_consistency_loss(output, target)

                total_loss = loss_l1 + loss_adv + loss_perceptual + loss_cycle
                total_loss.backward()
                self.optimizer_G.step()

                # Discriminator forward pass
                self.optimizer_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(target), torch.ones_like(self.discriminator(target)))
                fake_loss = self.adversarial_loss(self.discriminator(output.detach()), torch.zeros_like(self.discriminator(output.detach())))
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                if i % self.config['log_interval'] == 0:
                    print(f"Epoch [{epoch}/{self.config['epochs']}], Step [{i}/{len(self.train_loader)}], "
                          f"Loss: {total_loss.item():.4f}, D Loss: {d_loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for High-Resolution Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = HighResModelTrainer(config)
    trainer.load_data()
    trainer.train()
