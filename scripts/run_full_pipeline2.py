import itertools
import subprocess
import os
import pandas as pd
from datetime import datetime
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --- Experiment Parameters ---
BATCH_INDICES = [0, 100, 200, 300, 400]
BATCH_SIZES = [1, 2, 4, 8, 16]
NUM_ITERATIONS_RECONSTRUCTION = 1000 # Your existing iter count for reconstruction

SAVE_DIR_BASE = "saved_experiments"
CLIP_SUMMARY_CSV = "clip_experiment_summary.csv"

def run_pipeline_for_pair(class_a, class_b, repetition_idx, batch_size):
    experiment_name = f"{class_a}_vs_{class_b}"
    run_folder_name = f"run_{repetition_idx}" 

    print(f"\n=== Running pipeline: Pair={class_a}/{class_b}")

    exp_run_path = f"saved/{experiment_name}/{run_folder_name}/batch_size_{batch_size}"
    # Step 1: Capture gradient
    subprocess.run([
        sys.executable, "-m", "scripts.run_train_capture",
        f"data.class_a={class_a}",
        f"data.class_b={class_b}",
        f"experiment.name={experiment_name}",
        f"data.run_idx={repetition_idx}",
        f"data.capture_batch_idx={BATCH_INDICES}",
        f"data.batch_size={batch_size}"
    ], check=True)

    # Step 2: Reconstruct images
    subprocess.run([
        sys.executable, "-m", "scripts.run_reconstruction",
        f"experiment.name={experiment_name}",
        f"data.run_idx={repetition_idx}",
        f"data.capture_batch_idx={BATCH_INDICES}",
        f"data.class_a={class_a}",
        f"data.class_b={class_b}",
        f"reconstruction.num_iterations={NUM_ITERATIONS_RECONSTRUCTION}",
        f"data.batch_size={batch_size}"
    ], check=True)

    # Step 3: Classify with CLIP
    subprocess.run([
        sys.executable, "-m", "scripts.run_classify_clip",
        f"data.exp_name={experiment_name}",
        f"data.run_idx={repetition_idx}",
        f"data.capture_batch_idx={BATCH_INDICES}",
        f"data.class_a={class_a}",
        f"data.class_b={class_b}",
        f"data.batch_size={batch_size}"
    ], check=True)

    # Step 4: Classify originals with CLIP
    subprocess.run([
        sys.executable, "-m", "scripts.run_classify_clip",
        f"data.exp_name={experiment_name}",
        f"data.run_idx={repetition_idx}",
        f"data.capture_batch_idx={BATCH_INDICES}",
        f"data.class_a={class_a}",
        f"data.class_b={class_b}",
        f"data.original=True",
        f"data.batch_size={batch_size}"
    ], check=True)

    all_batch_results_for_this_run = []
    for batch_idx in BATCH_INDICES:
        clip_result_recon_data = {
            'successes': 0, 'selected_class_0': 'READ_ERROR', 'selected_class_1': 'READ_ERROR'
        }
        clip_result_original_data = {
            'successes': 0, 'selected_class_0': 'READ_ERROR', 'selected_class_1': 'READ_ERROR'
        }
        classify_clip_json_path_recon = exp_run_path + f"/clip_results/batch_{batch_idx}_recon.json"
        classify_clip_json_path_ori = exp_run_path + f"/clip_results/batch_{batch_idx}_original.json"

        with open(classify_clip_json_path_recon, 'r') as f:
            clip_result_recon_data = json.load(f)
        
        with open(classify_clip_json_path_ori, 'r') as f:
            clip_result_original_data = json.load(f)

        all_batch_results_for_this_run.append({
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
    return all_batch_results_for_this_run

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Only get RUN_IDX from cfg as requested
    RUN_IDX = cfg.pipeline.run_indices # <--- HERE IS THE ONLY THING FROM CFG IN MAIN

    # The other constants are still global as per your request
    # CIFAR10_CLASSES, BATCH_SIZES are used directly

    # Determine which repetition indices to run from the config/CLI override
    if isinstance(RUN_IDX, int): # Handle single integer from CLI
        repetition_indices_to_run = [RUN_IDX]
    else: # Assume it's a list from config or CLI override like "[5,6,7]"
        repetition_indices_to_run = list(RUN_IDX)

    class_pairs = list(itertools.combinations(CIFAR10_CLASSES, 2))
    
    total_combinations = len(class_pairs) * len(repetition_indices_to_run) * len(BATCH_SIZES)
    current_combination_num = 0

    print(f"Starting {total_combinations} total runs (pairs * repetitions * batch_sizes)")
    
    # This list will hold results for the current single run of this script
    current_repetition_results_buffer = []

    for class_a, class_b in class_pairs:
        for repetition_idx in repetition_indices_to_run: 
            # Reset buffer for each repetition_idx, ensuring separate file outputs
            current_repetition_results_buffer = [] 

            for batch_size in BATCH_SIZES: # Use global BATCH_SIZES
                current_combination_num += 1
                print(f"\n--- Progress (main loop): {current_combination_num}/{total_combinations} --- {datetime.now()}")

                # Call run_pipeline_for_pair without cfg argument
                results_for_this_batch_size = run_pipeline_for_pair(
                    class_a, class_b, repetition_idx, batch_size
                )
                current_repetition_results_buffer.extend(results_for_this_batch_size)

            # After all batch_sizes for a specific repetition_idx are done, save its CSV
            if current_repetition_results_buffer:
                df_repetition = pd.DataFrame(current_repetition_results_buffer)
                repetition_summary_csv_path = os.path.join(SAVE_DIR_BASE, f"clip_summary_run_{repetition_idx}.csv")
                df_repetition.to_csv(repetition_summary_csv_path, index=False)
                print(f"\nResults for repetition {repetition_idx} saved to: {repetition_summary_csv_path}")
                print(f"Total rows in clip_summary_run_{repetition_idx}.csv: {len(df_repetition)}")
            else:
                print(f"No results collected for repetition {repetition_idx}.")

    print("\n--- All individual repetition runs completed ---")

if __name__ == "__main__":
    os.makedirs(SAVE_DIR_BASE, exist_ok=True)
    main()
