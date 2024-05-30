# losses/cosine_similarity_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x, y):
        cos_sim = F.cosine_similarity(x, y)
        return 1 - cos_sim.mean()

if __name__ == "__main__":
    loss = CosineSimilarityLoss()
    print(loss)
