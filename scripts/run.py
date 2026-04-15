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
        print(f"Skipping redundant combination: {cfg.data.class_a} vs {cfg.data.class_b}")
        return

    exp_name = cfg.experiment.name
    class_a = cfg.data.class_a
    class_b = cfg.data.class_b
    repetition_idx = cfg.data.repetition
    batch_size = cfg.data.batch_size
    batch_indices = cfg.data.capture_batch_idx
    final_results_dir = cfg.paths.final_results_dir

    n_aggregate = cfg.data.get("n_aggregate", 1)

    print(f"\n=== Running pipeline: {class_a}/{class_b}, Rep={repetition_idx}, BS={batch_size}")

    if not cfg.skip.capture:
        run_capture_main(cfg)

    if not cfg.skip.reconstruct:
        run_reconstruction_main(cfg)

    if not cfg.skip.clip:
        cfg.data.original = False
        run_classify_clip_main(cfg)

        cfg.data.original = True
        run_classify_clip_main(cfg)

        agg_result_dict = {}

        all_results_rows = []

        # --- Process per-batch results inside the loop ---
        agg_counter = 0
        for i, batch_idx in enumerate(batch_indices):
            if cfg.data.aggregate and i%n_aggregate==0:
                clip_results_dir = cfg.paths.clip_results_dir
                classify_clip_json_path_recon_agg = os.path.join(clip_results_dir, f"recon_agg_{agg_counter}.json")
                classify_clip_json_path_ori_agg = os.path.join(clip_results_dir, f"original_agg_{agg_counter}.json")

                try:
                    with open(classify_clip_json_path_recon_agg, "r") as f:
                        clip_result_recon_data_agg = json.load(f)
                    with open(classify_clip_json_path_ori_agg, "r") as f:
                        clip_result_original_data_agg = json.load(f)

                    agg_result_dict = {
                        "recon_successes_agg": clip_result_recon_data_agg.get("successes", 0),
                        "recon_predicted_class_a_agg": clip_result_recon_data_agg.get(
                            "selected_class_0", "N/A"
                        ),
                        "recon_predicted_class_b_agg": clip_result_recon_data_agg.get(
                            "selected_class_1", "N/A"
                        ),
                        "ori_successes_agg": clip_result_original_data_agg.get("successes", 0),
                        "ori_predicted_class_a_agg": clip_result_original_data_agg.get(
                            "selected_class_0", "N/A"
                        ),
                        "ori_predicted_class_b_agg": clip_result_original_data_agg.get(
                            "selected_class_1", "N/A"
                        ),
                        "recon_success_0_agg": clip_result_recon_data_agg.get("success_0", 0),
                        "recon_success_1_agg": clip_result_recon_data_agg.get("success_1", 0),
                        "recon_majority_success_agg": clip_result_recon_data_agg.get("majority_success", 0),
                        "n_examples_0": clip_result_recon_data_agg.get("n_examples_0", 0),
                        "n_examples_1": clip_result_recon_data_agg.get("n_examples_1", 0),
                        "majority_class": clip_result_recon_data_agg.get("majority_class", 0),
                        
                    }

                except FileNotFoundError as e:
                    print(f"Aggregated results not found: {e}")
                    print("Skipping aggregation-related columns.")
                    agg_result_dict = {}  # Ensure dictionary is empty if files not found
                except Exception as e:
                    print(f"An error occurred while loading aggregated results: {e}")
                    agg_result_dict = {}

                agg_counter += 1

##################################
            clip_results_dir = cfg.paths.clip_results_dir
            classify_clip_json_path_recon = os.path.join(
                clip_results_dir, f"batch_{batch_idx}_recon.json"
            )
            classify_clip_json_path_ori = os.path.join(
                clip_results_dir, f"batch_{batch_idx}_original.json"
            )

            try:
                with open(classify_clip_json_path_recon, "r") as f:
                    clip_result_recon_data = json.load(f)
                with open(classify_clip_json_path_ori, "r") as f:
                    clip_result_original_data = json.load(f)

                # Create the base dictionary for the current batch
                results_dict = {
                    "class_a": class_a,
                    "class_b": class_b,
                    "repetition": repetition_idx,
                    "batch_idx": batch_idx,
                    "batch_size": batch_size,
                    "recon_successes": clip_result_recon_data.get("successes", 0),

                    "recon_success_0": clip_result_recon_data.get("success_0", "N/A"),
                    "recon_success_1": clip_result_recon_data.get("success_1", "N/A"),
                    "recon_majority_success": clip_result_recon_data.get("majority_success", "N/A"),
                    "recon_n_examples_0": clip_result_recon_data.get("n_examples_0", "N/A"),
                    "recon_n_examples_1": clip_result_recon_data.get("n_examples_1", "N/A"),
                    "recon_majority_class": clip_result_recon_data.get("majority_class", "N/A"),

                    "recon_predicted_class_a": clip_result_recon_data.get(
                        "selected_class_0", "N/A"
                    ),
                    "recon_predicted_class_b": clip_result_recon_data.get(
                        "selected_class_1", "N/A"
                    ),
                    "ori_successes": clip_result_original_data.get("successes", 0),

                    "ori_success_0": clip_result_original_data.get("success_0", "N/A"),
                    "ori_success_1": clip_result_original_data.get("success_1", "N/A"),
                    "ori_majority_success": clip_result_original_data.get("majority_success", "N/A"),
                    "ori_n_examples_0": clip_result_original_data.get("n_examples_0", "N/A"),
                    "ori_n_examples_1": clip_result_original_data.get("n_examples_1", "N/A"),
                    "ori_majority_class": clip_result_original_data.get("majority_class", "N/A"),

                    "ori_predicted_class_a": clip_result_original_data.get(
                        "selected_class_0", "N/A"
                    ),
                    "ori_predicted_class_b": clip_result_original_data.get(
                        "selected_class_1", "N/A"
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "aggregate_block_id": agg_counter, # Add block ID regardless of aggregation flag
                }
                
                # Merge the base dictionary with the aggregated results
                results_dict.update(agg_result_dict)
                all_results_rows.append(results_dict)

            except FileNotFoundError as e:
                print(f"Skipping batch {batch_idx}: {e}")
                continue

        df = pd.DataFrame(all_results_rows)
        os.makedirs(final_results_dir, exist_ok=True)
        csv_filename = f"{exp_name}_rep_{repetition_idx}_bs_{batch_size}.csv"
        df.to_csv(os.path.join(final_results_dir, csv_filename), index=False)


"""
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

            if cfg.data.aggregate:
                clip_results_dir = cfg.paths.clip_results_dir
                classify_clip_json_path_recon_agg = os.path.join(
                    clip_results_dir, "recon_agg.json"
                )
                classify_clip_json_path_ori_agg = os.path.join(
                    clip_results_dir, "original_agg.json"
                )
                with open(classify_clip_json_path_recon_agg, "r") as f:
                    clip_result_recon_data_agg = json.load(f)
                with open(classify_clip_json_path_ori_agg, "r") as f:
                    clip_result_original_data_agg = json.load(f)
                
            results_dict = {
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
            
            if cfg.data.aggregate:
                agg_result_dict = {
                    "recon_successes": clip_result_recon_data_agg.get("successes", 0),
                    "recon_predicted_class_a": clip_result_recon_data_agg.get(
                        "selected_class_0", "N/A"
                    ),
                    "recon_predicted_class_b": clip_result_recon_data_agg.get(
                        "selected_class_1", "N/A"
                    ),
                    "ori_successes": clip_result_original_data_agg.get("successes", 0),
                    "ori_predicted_class_a": clip_result_original_data_agg.get(
                        "selected_class_0", "N/A"
                    ),
                    "ori_predicted_class_b": clip_result_original_data_agg.get(
                        "selected_class_1", "N/A"
                    )
                }
                results_dict = results_dict | agg_result_dict
            all_results_rows.append(results_dict)

        df = pd.DataFrame(all_results_rows)

        os.makedirs(final_results_dir, exist_ok=True)
        csv_filename = f"{exp_name}_rep_{repetition_idx}_bs_{batch_size}.csv"
        df.to_csv(os.path.join(final_results_dir, csv_filename), index=False)
        results_dict = {
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

"""

if __name__ == "__main__":
    main()
