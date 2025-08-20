from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


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


def smart_evaluation(reconstructed: torch.Tensor, true_images: torch.Tensor, labels: torch.Tensor):
    recon_np = reconstructed.cpu().numpy()
    true_np = true_images.cpu().numpy()
    labels = labels.cpu().numpy()

    # Rearrange to [B, H, W, C] for skimage
    recon_np = np.transpose(recon_np, (0, 2, 3, 1))
    true_np = np.transpose(true_np, (0, 2, 3, 1))

    total_ssims = []

    for label in np.unique(labels):
        true_group = true_np[labels == label]
        recon_group = recon_np[labels == label]

        n = len(true_group)
        assert len(recon_group) == n, "Mismatch in group sizes"

        ssim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ssim_matrix[i, j] = ssim(
                    true_group[i], recon_group[j], data_range=1.0, channel_axis=-1
                )
        row_ind, col_ind = linear_sum_assignment(-ssim_matrix)

        total_ssims.extend(ssim_matrix[row_ind, col_ind])

    return float(np.mean(total_ssims))
