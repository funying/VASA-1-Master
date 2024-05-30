# utils/checkpoint.py

import torch

def save_checkpoint(model, optimizer, epoch, path):
    """Saves the model and optimizer state to a checkpoint file."""
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, path)

def load_checkpoint(model, optimizer, path, device):
    """Loads the model and optimizer state from a checkpoint file."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    epoch = state['epoch']
    return model, optimizer, epoch

if __name__ == "__main__":
    from models.student_model import StudentModel
    model = StudentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    save_checkpoint(model, optimizer, 0, 'checkpoint.pth')
    model, optimizer, epoch = load_checkpoint(model, optimizer, 'checkpoint.pth', torch.device('cpu'))
    print(f'Checkpoint loaded from epoch {epoch}')
