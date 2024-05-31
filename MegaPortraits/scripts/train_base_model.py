# scripts/train_base_model.py

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
from models.appearance_encoder import AppearanceEncoder
from models.motion_encoder import MotionEncoder
from models.warping_generators import WarpingGenerator
from models.conv3d import Conv3D
from models.conv2d import Conv2D
from models.discriminator import PatchGANDiscriminator
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial_loss import AdversarialLoss
from losses.cycle_consistency_loss import CycleConsistencyLoss
from losses.pairwise_loss import PairwiseLoss
from losses.cosine_similarity_loss import CosineSimilarityLoss
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from datasets.dataset import MegaPortraitDataset

class BaseModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.appearance_encoder = AppearanceEncoder().to(self.device)
        self.motion_encoder = MotionEncoder().to(self.device)
        self.warping_generator_s = WarpingGenerator().to(self.device)
        self.warping_generator_d = WarpingGenerator().to(self.device)
        self.conv3d = Conv3D().to(self.device)
        self.conv2d = Conv2D().to(self.device)

        self.discriminator = PatchGANDiscriminator().to(self.device)

        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.adversarial_loss = AdversarialLoss().to(self.device)
        self.cycle_consistency_loss = CycleConsistencyLoss().to(self.device)
        self.pairwise_loss = PairwiseLoss().to(self.device)
        self.cosine_similarity_loss = CosineSimilarityLoss().to(self.device)

        self.optimizer_G = optim.AdamW(
            list(self.appearance_encoder.parameters()) +
            list(self.motion_encoder.parameters()) +
            list(self.warping_generator_s.parameters()) +
            list(self.warping_generator_d.parameters()) +
            list(self.conv3d.parameters()) +
            list(self.conv2d.parameters()),
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

                # Extract appearance and motion features
                v_s = self.appearance_encoder(source)
                e_s = self.motion_encoder(source)
                R_s, t_s, z_s = e_s
                v_d = self.appearance_encoder(target)
                e_d = self.motion_encoder(target)
                R_d, t_d, z_d = e_d

                # Generate warping fields
                w_s = self.warping_generator_s(R_s, t_s, z_s, e_s)
                w_d = self.warping_generator_d(R_d, t_d, z_d, e_d)

                # Process volumetric features
                v_s_warped = self.conv3d(w_s)
                v_d_warped = self.conv3d(w_d)

                # Generate final output
                output = self.conv2d(v_s_warped)

                # Calculate losses
                loss_perceptual = self.perceptual_loss(output, target)
                loss_adv = self.adversarial_loss(self.discriminator(target), self.discriminator(output))
                loss_cycle = self.cycle_consistency_loss(output, target)
                loss_pairwise = self.pairwise_loss(v_s, v_d)
                loss_cosine = self.cosine_similarity_loss(e_s, e_d)

                total_loss = loss_perceptual + loss_adv + loss_cycle + loss_pairwise + loss_cosine
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
    parser = argparse.ArgumentParser(description='Training script for Base Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = BaseModelTrainer(config)
    trainer.load_data()
    trainer.train()
