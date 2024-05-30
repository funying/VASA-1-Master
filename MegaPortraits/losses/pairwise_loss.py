# losses/pairwise_loss.py

import torch
import torch.nn as nn

class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, I_ij_pose, I_ji_dyn):
        return self.l1_loss(I_ij_pose, I_ji_dyn)

if __name__ == "__main__":
    loss = PairwiseLoss()
    print(loss)
