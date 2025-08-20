import os
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import CLIPProcessor, CLIPModel

from src.inversion import reconstruct
from src.utils import Batch_data, cifar10_class_names


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = cfg.paths.clip_base_path

    model = CLIPModel.from_pretrained(model_path).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)

    exp_name = cfg.experiment.name
    exp_dir = os.path.join(cfg.experiment.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    run_idx = cfg.data.repetition
    batch_size = cfg.data.batch_size
    reconstruction_results_dir = cfg.paths.reconstruction_results_dir
    input_path = cfg.paths.batch_data_dir
    os.makedirs(reconstruction_results_dir, exist_ok=True)

    # guidence
    guidance = [-1, -1]
    if cfg.reconstruction.oracle_guidance:
        class_a_index = cifar10_class_names.index(cfg.data.class_a)
        class_b_index = cifar10_class_names.index(cfg.data.class_b)
        guidance = [class_a_index, class_b_index]

    batch_idx = cfg.data.capture_batch_idx

    for idx in batch_idx:
        batch_data = Batch_data()
        batch_data.load(input_path, idx, skip=False)
        # batch_data.remove(input_path, idx)

        for rerun_n in range(cfg.reconstruction.reruns):
            rerun_output_path = os.path.join(
                reconstruction_results_dir, f"rerun_{rerun_n}"
            )
            os.makedirs(rerun_output_path, exist_ok=True)

            if cfg.wandb_log:
                wandb_config_dict = OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=False
                )
                wandb.init(
                    project="cifar10-gradient-inversion",
                    group=cfg.group_name,
                    name=f"{exp_name}/run_{run_idx}/batch_size_{batch_size}/batch_{idx}/rerun_{rerun_n}",
                    config=wandb_config_dict,
                    dir=os.path.join("saved_experiments", exp_name),
                )

            reconstruct(
                batch_data=batch_data,
                output_path=rerun_output_path,
                batch_idx=idx,
                device=device,
                num_iterations=cfg.reconstruction.num_iterations,
                tv_coeff=cfg.reconstruction.tv_coeff,
                clip_coeff=cfg.reconstruction.clip_coeff,
                lr=cfg.reconstruction.lr,
                wandb_log=cfg.wandb_log,
                only_first=cfg.data.only_first_layer,
                guidance=guidance,
                clip_model=model,
                clip_processor=processor,
            )
            if cfg.wandb_log:
                wandb.finish()


if __name__ == "__main__":
    main()
