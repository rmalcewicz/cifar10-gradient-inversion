from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from typing import List, Tuple

def evaluate_reconstruction(reconstructed: torch.Tensor, true_images: torch.Tensor):
    """
    Compute average PSNR and SSIM between reconstructed and true images.

    Args:
        reconstructed: Tensor of shape [B, C, H, W]
        true_images:   Tensor of same shape [B, C, H, W]

    Returns:
        Dictionary with average PSNR and SSIM.
    """
    recon_np = reconstructed.cpu().numpy()
    true_np = true_images.cpu().numpy()

    # Rearrange to [B, H, W, C] for skimage
    recon_np = np.transpose(recon_np, (0, 2, 3, 1))
    true_np = np.transpose(true_np, (0, 2, 3, 1))

    psnr_scores = [psnr(t, r, data_range=1.0) for t, r in zip(true_np, recon_np)]
    ssim_scores = [ssim(t, r, data_range=1.0, channel_axis=-1) for t, r in zip(true_np, recon_np)]

    return {
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
    }

def predict_with_clip(
    images: List[torch.Tensor],
    class_names: List[str],
    model_name="openai/clip-vit-base-patch32",
    true_labels: List[str] = None
) -> Tuple[List[str], float]:
    
    """
    Predicts class labels for a list of images using CLIP.

    Args:
        images (List[Tensor]): List of reconstructed images (C,H,W) as Tensors.
        class_names (List[str]): E.g., ["airplane", "automobile"]
        model_name (str): HuggingFace model name for CLIP.

    Returns:
        predictions (List[str]): Predicted class for each image.
        accuracy (float): Accuracy over the provided images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    prompts = [f"a photo of a {c}" for c in class_names]

    pil_images = [transforms.ToPILImage()(img_tensor.cpu()) for img_tensor in images]

    inputs = processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=-1)

    predictions = []
    correct = 0
    for i, prob in enumerate(probs):
        top_idx = prob.argmax().item()
        pred_class = class_names[top_idx]
        predictions.append(pred_class)

        true_class = class_names[i % 2]
        if pred_class == true_class:
            correct += 1

    if true_labels:
        correct = sum(pred == true for pred, true in zip(predictions, true_labels))
        accuracy = correct / len(images)

    return predictions, accuracy