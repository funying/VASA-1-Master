import torch

def save_intermediate(tensor, path, chunk_id):
    torch.save(tensor, f"{path}/chunk_{chunk_id}.pth")

def load_intermediate(path, chunk_id, device):
    return torch.load(f"{path}/chunk_{chunk_id}.pth", map_location=device)
