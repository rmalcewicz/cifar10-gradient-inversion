import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.inversion import reconstruct
import torchvision.utils as vutils
import os


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "results/exp1"
    os.makedirs(output_dir, exist_ok=True)

    reconstructed = reconstruct(
        device=device,
        lr=0.01,
        iterations=4000,
        tv_weight=1e-2,
        grad_path="saved/exp2"
    )
    # Save image grid
    vutils.save_image(reconstructed, f"{output_dir}/reconstructed.png", nrow=8, normalize=True)
    print(f"Reconstruction saved to {output_dir}/reconstructed.png")

import os
import hydra
import torch
import wandb
from omegaconf import DictConfig

from src.inversion import reconstruct
from src.utils import load_target_data

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = cfg.experiment.name
    exp_dir = os.path.join(cfg.experiment.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    wandb.innit(
        project="cifar10-gradient-inversion",
        name=exp_name,
        config=cfg,
        dir=exp_dir
    )

    recon = reconstruct(
        grad_path=exp_dir,
        device=device,
        num_iterations=cfg.reconstruction.num_iterations,
        tv_coeff=cfg.reconstruction.tv_coeff,
        lr=cfg.reconstruction.lr,
        wandb_log = True
    )