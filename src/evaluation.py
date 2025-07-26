from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch

def evaluate_reconstruction(reconstructed: torch.Tensor, true_images: torch.Tensor):
    """
    Compute average PSNR and SSIM between reconstructed and true images.

    Args:
        reconstructed: Tensor of shape [B, C, H, W]
        true_images:   Tensor of same shape [B, C, H, W]

    Returns:
        Dictionary with average PSNR and SSIM.
    """
    recon_np = reconstructed.cpu().numpy()
    true_np = true_images.cpu().numpy()

    # Rearrange to [B, H, W, C] for skimage
    recon_np = np.transpose(recon_np, (0, 2, 3, 1))
    true_np = np.transpose(true_np, (0, 2, 3, 1))

    psnr_scores = [psnr(t, r, data_range=1.0) for t, r in zip(true_np, recon_np)]
    ssim_scores = [ssim(t, r, data_range=1.0, channel_axis=-1) for t, r in zip(true_np, recon_np)]

    return {
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
    }
