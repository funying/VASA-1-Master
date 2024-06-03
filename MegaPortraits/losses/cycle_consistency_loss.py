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

        # Process in chunks
        chunk_size = 25  
        losses = []
        for i in range(0, x.size(0), chunk_size):
            x_chunk = x[i:i+chunk_size]
            y_chunk = y[i:i+chunk_size]
            loss = self.l1_loss(x_chunk, y_chunk)
            losses.append(loss.item())
        
        return sum(losses) / len(losses)

if __name__ == "__main__":
    loss = CycleConsistencyLoss()
    print(loss)
