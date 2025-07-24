import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.train_capture import capture_batch

device = "cuda" if torch.cuda.is_available() else "cpu"
capture_batch(device, class_a = "cat", class_b = "dog", experiment_name="exp1", batch_size=1)