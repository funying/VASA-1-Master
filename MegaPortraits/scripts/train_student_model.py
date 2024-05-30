# scripts/train_student_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.student_model import StudentModel
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial_loss import AdversarialLoss

class StudentModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = StudentModel().to(self.device)
        self.discriminator = PatchGANDiscriminator().to(self.device)

        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.adversarial_loss = AdversarialLoss().to(self.device)

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
        # Assuming dataset.py provides the dataset class
        from datasets.dataset import MegaPortraitDataset
        train_dataset = MegaPortraitDataset(self.config['data_path'], transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)

    def train(self):
        for epoch in range(self.config['epochs']):
            for i, data in enumerate(self.train_loader):
                source, target = data
                source = source.to(self.device)
                target = target.to(self.device)

                # Generator forward pass
                self.optimizer_G.zero_grad()

                output = self.model(source)

                # Calculate losses
                loss_perceptual = self.perceptual_loss(output, target)
                loss_adv = self.adversarial_loss(self.discriminator(target), self.discriminator(output))

                total_loss = loss_perceptual + loss_adv
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
    import yaml
    with open('configs/training/student_model.yaml', 'r') as f:
        config = yaml.safe_load(f)

    trainer = StudentModelTrainer(config)
    trainer.load_data()
    trainer.train()
