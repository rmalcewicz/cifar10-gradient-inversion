import os
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import json

from src.inversion import reconstruct
from src.utils import load_target_data

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = cfg.experiment.name
    exp_dir = os.path.join(cfg.experiment.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    wandb_config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    
    wandb.init(
        project="cifar10-gradient-inversion",
        name=exp_name,
        config=wandb_config_dict,
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

if __name__ == "__main__":
    main()