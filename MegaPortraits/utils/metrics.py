# utils/metrics.py

import torch
import torch.nn.functional as F

def calculate_metrics(source, target, generated):
    """Calculates evaluation metrics for generated images."""
    mse = F.mse_loss(generated, target).item()
    psnr = 10 * torch.log10(1 / mse).item()
    ssim = calculate_ssim(generated, target).item()
    return {'MSE': mse, 'PSNR': psnr, 'SSIM': ssim}

def calculate_ssim(img1, img2):
    """Calculates the Structural Similarity Index (SSIM) between two images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1 * mu1
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2 * mu2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

if __name__ == "__main__":
    img1 = torch.rand(1, 3, 224, 224)
    img2 = torch.rand(1, 3, 224, 224)
    metrics = calculate_metrics(img1, img2, img1)
    print(metrics)
