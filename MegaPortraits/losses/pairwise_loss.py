# losses/pairwise_loss.py

import torch
import torch.nn as nn
import os

class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, I_ij_pose, I_ji_dyn):
        return self.l1_loss(I_ij_pose, I_ji_dyn)

    def calculate_and_save(self, v_s_chunk, v_d_chunk, chunk_index, intermediate_path):
        I_ij_pose, I_ji_dyn = self.calculate(v_s_chunk, v_d_chunk)
        torch.save(I_ij_pose.cpu(), os.path.join(intermediate_path, f'I_ij_pose_{chunk_index}.pt'))
        torch.save(I_ji_dyn.cpu(), os.path.join(intermediate_path, f'I_ji_dyn_{chunk_index}.pt'))
        del I_ij_pose, I_ji_dyn
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def calculate(self, v_s_chunk, v_d_chunk):
        # Placeholder for actual calculation logic
        I_ij_pose = v_s_chunk
        I_ji_dyn = v_d_chunk
        return I_ij_pose, I_ji_dyn

if __name__ == "__main__":
    loss = PairwiseLoss()
    print(loss)
