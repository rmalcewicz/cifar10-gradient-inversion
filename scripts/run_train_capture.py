import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.train_capture import capture_batch

device = "cuda" if torch.cuda.is_available() else "cpu"
capture_batch(device, class_a = "cat", class_b = "dog", experiment_name="exp2", batch_size=2)

import os
import hydra
import torch
from omegaconf import DictConfig

from src.train_capture import capture_batch

@hydra.main(config_path="../configs", config_name="config")

def main(cfg:DictConfig):
    device = "cuda" if torch.cuda.is_avaliable() else "cpu"
    print(f"Running on {device}")

    exp_dir = os.path.join(cfg.experiment.save_dir, cfg.experiment.name)
    os.makedirs(exp_dir, exist_ok=True)

    capture_batch(
        device=device,
        class_a=cfg.data.class_a,
        class_b=cfg.data.class_b,
        experiment_name=cfg.experiment.name,
        batch_size=cfg.data.batch_size
    )

if __name__ == "__main__":
    main()
