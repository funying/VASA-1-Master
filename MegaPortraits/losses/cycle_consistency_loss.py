# losses/cycle_consistency_loss.py

import torch
import torch.nn as nn

class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        return self.l1_loss(x, y)

if __name__ == "__main__":
    loss = CycleConsistencyLoss()
    print(loss)
