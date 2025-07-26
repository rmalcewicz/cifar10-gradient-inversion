import torch
import os

def load_target_data(exp_dir, device):
    """
    Loads original target data saved during capture
    Returns (label_tensor, image tensor)
    """

    target_path = f"{exp_dir}/target.pt"
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"No target.pt in {exp_dir}")
    
    target = torch.load(target_path, map_location=device)
    label = target['label']
    image = target['image']
    return label, image

def TVLoss(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w