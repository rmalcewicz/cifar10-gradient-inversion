import hydra
import pandas as pd
import os
import json
from omegaconf import DictConfig
from datetime import datetime


from scripts.run_train_capture import main as run_capture_main
from scripts.run_reconstruction import main as run_reconstruction_main
from scripts.run_classify_clip import main as run_classify_clip_main


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.data.class_a >= cfg.data.class_b:
        print(
            f"Skipping redundant combination: {cfg.data.class_a} vs {cfg.data.class_b}"
        )
        return

    exp_name = cfg.experiment.name
    class_a = cfg.data.class_a
    class_b = cfg.data.class_b
    repetition_idx = cfg.data.repetition
    batch_size = cfg.data.batch_size
    batch_indices = cfg.data.capture_batch_idx
    final_results_dir = cfg.paths.final_results_dir

    print(
        f"\n=== Running pipeline: {class_a}/{class_b}, Rep={repetition_idx}, BS={batch_size}"
    )

    if not cfg.skip.capture:
        run_capture_main(cfg)

    if not cfg.skip.reconstruct:
        run_reconstruction_main(cfg)

    if not cfg.skip.clip:
        cfg.data.original = False
        run_classify_clip_main(cfg)

        cfg.data.original = True
        run_classify_clip_main(cfg)

        all_results_rows = []
        for batch_idx in batch_indices:
            clip_results_dir = cfg.paths.clip_results_dir
            classify_clip_json_path_recon = os.path.join(
                clip_results_dir, f"batch_{batch_idx}_recon.json"
            )
            classify_clip_json_path_ori = os.path.join(
                clip_results_dir, f"batch_{batch_idx}_original.json"
            )

            with open(classify_clip_json_path_recon, "r") as f:
                clip_result_recon_data = json.load(f)
            with open(classify_clip_json_path_ori, "r") as f:
                clip_result_original_data = json.load(f)

            all_results_rows.append(
                {
                    "class_a": class_a,
                    "class_b": class_b,
                    "repetition": repetition_idx,
                    "batch_idx": batch_idx,
                    "batch_size": batch_size,
                    "recon_successes": clip_result_recon_data.get("successes", 0),
                    "recon_predicted_class_a": clip_result_recon_data.get(
                        "selected_class_0", "N/A"
                    ),
                    "recon_predicted_class_b": clip_result_recon_data.get(
                        "selected_class_1", "N/A"
                    ),
                    "ori_successes": clip_result_original_data.get("successes", 0),
                    "ori_predicted_class_a": clip_result_original_data.get(
                        "selected_class_0", "N/A"
                    ),
                    "ori_predicted_class_b": clip_result_original_data.get(
                        "selected_class_1", "N/A"
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        df = pd.DataFrame(all_results_rows)

        os.makedirs(final_results_dir, exist_ok=True)
        csv_filename = f"{exp_name}_rep_{repetition_idx}_bs_{batch_size}.csv"
        df.to_csv(os.path.join(final_results_dir, csv_filename), index=False)


if __name__ == "__main__":
    main()
