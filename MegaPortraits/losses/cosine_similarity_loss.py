# losses/cosine_similarity_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, e_s, e_d):
        if isinstance(e_s, tuple) and isinstance(e_d, tuple):
            cos_sim = 0
            for tensor_s, tensor_d in zip(e_s, e_d):
                cos_sim += F.cosine_similarity(tensor_s, tensor_d, dim=1).mean()
            return -cos_sim / len(e_s)
        else:
            return -F.cosine_similarity(e_s, e_d, dim=1).mean()

if __name__ == "__main__":
    loss = CosineSimilarityLoss()
    print(loss)
