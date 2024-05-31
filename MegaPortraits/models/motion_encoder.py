# models/motion_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.head_pose = resnet18(pretrained=True)
        self.expression = resnet18(pretrained=True)
        self.additional_feature = resnet18(pretrained=True)
        
        self.head_pose.fc = nn.Linear(self.head_pose.fc.in_features, 6)  # Output for rotations and translations
        self.expression.fc = nn.Linear(self.expression.fc.in_features, 50)  # Latent expression descriptors
        self.additional_feature.fc = nn.Linear(self.additional_feature.fc.in_features, 20)  # Another set of features

    def forward(self, x):
        pose = self.head_pose(x)
        expr = self.expression(x)
        add_feat = self.additional_feature(x)
        return pose, expr, add_feat

if __name__ == "__main__":
    model = MotionEncoder()
    print(model)
