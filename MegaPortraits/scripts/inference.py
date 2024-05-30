# scripts/inference.py

import torch
from torchvision import transforms
from PIL import Image
from models.student_model import StudentModel

class Inference:
    def __init__(self, model_path, device):
        self.device = device
        self.model = StudentModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        return image

    def generate(self, source_path, driving_path):
        source_image = self.preprocess(source_path)
        driving_image = self.preprocess(driving_path)
        with torch.no_grad():
            generated_image = self.model(source_image, driving_image)
        return generated_image.squeeze(0).cpu()

    def save_image(self, tensor, path):
        image = transforms.ToPILImage()(tensor)
        image.save(path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inference script for MegaPortraits')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source image')
    parser.add_argument('--driving_path', type=str, required=True, help='Path to the driving image')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference = Inference(args.model_path, device)
    generated_image = inference.generate(args.source_path, args.driving_path)
    inference.save_image(generated_image, args.output_path)
    print(f"Generated image saved at {args.output_path}")
