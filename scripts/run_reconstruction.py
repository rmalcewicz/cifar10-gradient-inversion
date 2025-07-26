import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.inversion import reconstruct
import torchvision.utils as vutils
import os


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "results/exp1"
    os.makedirs(output_dir, exist_ok=True)

    reconstructed = reconstruct(
        device=device,
        lr=0.01,
        iterations=4000,
        tv_weight=1e-2,
        grad_path="saved/exp2"
    )
    # Save image grid
    vutils.save_image(reconstructed, f"{output_dir}/reconstructed.png", nrow=8, normalize=True)
    print(f"Reconstruction saved to {output_dir}/reconstructed.png")

if __name__ == "__main__":
    main()