import hydra
import pandas as pd
import os
import json
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import wandb

# Import the main functions of the sub-scripts
from scripts.run_train_capture import main as run_capture_main
from scripts.run_reconstruction import main as run_reconstruction_main
from scripts.run_classify_clip import main as run_classify_clip_main

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.data.class_a >= cfg.data.class_b:
        print(f"Skipping redundant combination: {cfg.data.class_a} vs {cfg.data.class_b}")
        return
    
    exp_name = cfg.experiment.name
    class_a = cfg.data.class_a
    class_b = cfg.data.class_b
    repetition_idx = cfg.data.repetition
    batch_size = cfg.data.batch_size
    batch_indices = cfg.data.capture_batch_idx

    print(f"\n=== Running pipeline: {class_a}/{class_b}, Rep={repetition_idx}, BS={batch_size}")


    run_capture_main(cfg)
    metrics = run_reconstruction_main(cfg)

    cfg.data.original = False
    run_classify_clip_main(cfg)

    cfg.data.original = True
    run_classify_clip_main(cfg)

    all_results_rows = []
    exp_run_path = f"saved/{exp_name}/run_{repetition_idx}/batch_size_{batch_size}"
    for batch_idx in batch_indices:
        classify_clip_json_path_recon = exp_run_path + f"/clip_results/batch_{batch_idx}_recon.json"
        classify_clip_json_path_ori = exp_run_path + f"/clip_results/batch_{batch_idx}_original.json"

        # ... (Rest of your JSON loading and aggregation code)
        with open(classify_clip_json_path_recon, 'r') as f:
            clip_result_recon_data = json.load(f)
        with open(classify_clip_json_path_ori, 'r') as f:
            clip_result_original_data = json.load(f)

        all_results_rows.append({
            "class_a": class_a,
            "class_b": class_b,
            "repetition": repetition_idx,
            "batch_idx": batch_idx,
            "batch_size": batch_size,

            "recon_successes": clip_result_recon_data.get('successes', 0),
            "recon_predicted_class_a": clip_result_recon_data.get('selected_class_0', 'N/A'),
            "recon_predicted_class_b": clip_result_recon_data.get('selected_class_1', 'N/A'),

            "ori_successes": clip_result_original_data.get('successes', 0),
            "ori_predicted_class_a": clip_result_original_data.get('selected_class_0', 'N/A'),
            "ori_predicted_class_b": clip_result_original_data.get('selected_class_1', 'N/A'),

            "timestamp": datetime.now().isoformat()
        })
    
    # 4. Save results to a CSV specific to this run
    SAVE_DIR_BASE = "saved_experiments/exp2"
    os.makedirs(SAVE_DIR_BASE, exist_ok=True)
    df = pd.DataFrame(all_results_rows)
    df.to_csv(os.path.join(SAVE_DIR_BASE, f"{exp_name}_rep_{repetition_idx}_bs_{batch_size}.csv"), index=False)

if __name__ == "__main__":
    main()