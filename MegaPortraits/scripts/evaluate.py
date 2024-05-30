# scripts/evaluate.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.student_model import StudentModel
from datasets.dataset import MegaPortraitDataset
from utils.metrics import calculate_metrics

class Evaluator:
    def __init__(self, model_path, data_path, device):
        self.device = device
        self.model = StudentModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        self.data_loader = self.load_data(data_path)

    def load_data(self, data_path):
        dataset = MegaPortraitDataset(data_path, transform=self.transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        return data_loader

    def evaluate(self):
        total_metrics = {}
        for i, data in enumerate(self.data_loader):
            source, driving = data
            source = source.to(self.device)
            driving = driving.to(self.device)
            with torch.no_grad():
                generated_image = self.model(source, driving)
            metrics = calculate_metrics(source, driving, generated_image)
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)

        avg_metrics = {key: sum(values) / len(values) for key, values in total_metrics.items()}
        return avg_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluation script for MegaPortraits')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset for evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(args.model_path, args.data_path, device)
    metrics = evaluator.evaluate()
    print(f"Evaluation metrics: {metrics}")
