# losses/cycle_consistency_loss.py

import torch
import torch.nn as nn

class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        # Ensure the target tensor is resized to match the output tensor's shape
        if x.shape != y.shape:
            y = torch.nn.functional.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.l1_loss(x, y)

if __name__ == "__main__":
    loss = CycleConsistencyLoss()
    print(loss)
