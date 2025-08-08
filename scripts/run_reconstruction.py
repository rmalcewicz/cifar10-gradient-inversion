import os
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.inversion import reconstruct
from src.utils import Batch_data

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = cfg.experiment.name
    exp_dir = os.path.join(cfg.experiment.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    run_idx = cfg.data.repetition
    batch_size = cfg.data.batch_size
    output_path = cfg.paths.reconstruction_results_dir
    input_path = cfg.paths.batch_data_dir
    os.makedirs(output_path, exist_ok=True)

    batch_idx = cfg.data.capture_batch_idx

    for idx in batch_idx:

        batch_data = Batch_data()
        batch_data.load(input_path, idx, skip = False)
        batch_data.remove(input_path, idx)

        wandb_config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        if cfg.wandb_log:
            wandb.init(
                project="cifar10-gradient-inversion",
                name=f"{exp_name}/run_{run_idx}/batch_size_{batch_size}/batch_{idx}",
                config=wandb_config_dict,
                dir=exp_dir
            )

        recon = reconstruct(
            batch_data=batch_data,
            output_path=output_path,
            batch_idx=idx,
            device=device,
            num_iterations=cfg.reconstruction.num_iterations,
            tv_coeff=cfg.reconstruction.tv_coeff,
            lr=cfg.reconstruction.lr,
            wandb_log = cfg.wandb_log
        )
        if cfg.wandb_log:
            wandb.finish()

if __name__ == "__main__":
    main()