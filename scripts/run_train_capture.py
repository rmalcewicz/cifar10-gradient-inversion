import os
import hydra
import torch
from omegaconf import DictConfig

from src.train_capture import capture_batch

@hydra.main(config_path="../configs", config_name="config", version_base=None)

def main(cfg:DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"Running on {device}")

    exp_name = cfg.experiment.name
    run_idx = cfg.data.repetition
    batch_size = cfg.data.batch_size

    output_path = cfg.batch_data_dir
    os.makedirs(output_path, exist_ok=True)

    capture_batch(
        device=device,
        class_a=cfg.data.class_a,
        class_b=cfg.data.class_b,
        experiment_name=cfg.experiment.name,
        output_path=output_path,
        batch_size=cfg.data.batch_size,
        capture_batch_idx=cfg.data.capture_batch_idx
    )

if __name__ == "__main__":
    main()
