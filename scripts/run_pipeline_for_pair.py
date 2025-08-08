import itertools
import subprocess
import os
import pandas as pd
from datetime import datetime

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

SAVE_DIR = "saved"
RECON_DIR = "reconstructed"
CLIP_CSV_PATH = "clip_accuracy_summary.csv"

def run_pipeline_for_pair(class_a, class_b):
    experiment_name = f"{class_a}_vs_{class_b}_v3"
    iters = 4000
    print(f"\n=== Running experiment: {experiment_name} ===")

    # Step 1: Capture gradient
    subprocess.run([
        "python", "-m", "scripts.run_train_capture",
        f"data.class_a={class_a}",
        f"data.class_b={class_b}",
        f"experiment.name={experiment_name}"
    ], check=True)

    # Step 2: Reconstruct images
    subprocess.run([
        "python", "-m", "scripts.run_reconstruction",
        f"experiment.name={experiment_name}",
        f"data.class_a={class_a}",
        f"data.class_b={class_b}",
        f"reconstruction.num_iterations={iters}"
    ], check=True)

    # Step 3: Classify with CLIP
    subprocess.run([
        "python", "-m", "scripts.run_classify_clip",
        f"data.exp_name={experiment_name}",
        f"data.class_a={class_a}",
        f"data.class_b={class_b}"
    ], check=True)

    subprocess.run([
        "python", "-m", "scripts.run_classify_clip",
        f"data.exp_name={experiment_name}",
        f"data.class_a={class_a}",
        f"data.class_b={class_b}",
        f"data.original=True"
    ], check=True)

def main():
    run_pipeline_for_pair("cat", "dog")

if __name__ == "__main__":
    main()