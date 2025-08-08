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



class Batch_data:

    def __init__(self):
        self.gradient = None
        self.images = None
        self.labels = None
        self.model_state = None
    
    def load(self, batch_path, batch_n, skip=True):
        target_path = f"{batch_path}/batch_{batch_n}"
        if not skip:
            self.gradient = torch.load(f"{target_path}/batch_gradient.pt")
            self.model_state = torch.load(f"{target_path}/model_state.pt")
        self.images = torch.load(f"{target_path}/batch_images.pt")
        self.labels = torch.load(f"{target_path}/batch_labels.pt")
        
    
    def remove(self, batch_path, batch_n):
        target_path = f"{batch_path}/batch_{batch_n}"
        os.remove(f"{target_path}/batch_gradient.pt")
        os.remove(f"{target_path}/model_state.pt")