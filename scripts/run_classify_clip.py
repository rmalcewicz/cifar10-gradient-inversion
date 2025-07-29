import torch
import glob
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import os
import csv
from datetime import datetime

from src.utils import Batch_data

@hydra.main(config_path="../configs", config_name="classify_clip", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = cfg.clip.model
    if model_name == "ViT-B/32":
        model_name = "openai/clip-vit-base-patch32"
    elif model_name == "ViT-L/14":
        model_name = "openai/clip-vit-large-patch14"

    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name)


    all_similarities_df_rows = []
    all_probabilities_df_rows = []
    all_img_names = []
    list_of_prompts = OmegaConf.to_container(cfg.clip.prompts, resolve=True)

    batch_idx = cfg.data.capture_batch_idx
    exp_name = cfg.data.exp_name

    class_a = cfg.data.class_a
    class_b = cfg.data.class_b
    
    for idx in batch_idx:
        print(f"\n Processing {exp_name} batch_{idx} ...")

        batch_file = f"saved/{exp_name}/results/recons/batch_{idx}.pt"
        batch_data = Batch_data()
        batch_data.load(exp_name, idx)
        labels = batch_data.labels
        
        if not os.path.exists(batch_file):
            print(f"Batch file not found: {batch_file}. Skipping this batch.")
            continue
        try:
            images_tensor_batch = torch.load(batch_file)
        except Exception as e:
            print(f"Error loading {batch_file}: {e}. Skipping this batch.")
            continue

        images_tensor_batch = torch.load(batch_file)

        pil_images = [transforms.ToPILImage()(img_tensor.cpu()) for img_tensor in images_tensor_batch]
        inputs = processor(text=list_of_prompts, images=pil_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            similarity_scores = outputs.logits_per_image
 
            similarity_probabilities = similarity_scores.softmax(dim=-1)
        
        for i in range(len(pil_images)):
            image_name_for_df = f"{os.path.basename(batch_file)}_img_{i}"
            all_img_names.append(image_name_for_df)
            all_similarities_df_rows.append(similarity_scores[i].cpu().numpy())
            all_probabilities_df_rows.append(similarity_probabilities[i].cpu().numpy())
        
        print(f"\n--- Predictions for {batch_file} ---")    
        for i, probs in enumerate(similarity_probabilities):
            top_prob, top_label = probs.topk(1)
            predicted_class = list_of_prompts[top_label].replace("a photo of a ", "")
            print(f"Image {i}: {predicted_class} ({top_prob.item() * 100:.2f}%)")

    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.width', 1000)        # Adjust width to prevent wrapping (experiment with this value)
    pd.set_option('display.float_format', '{:.4f}'.format) # Optional: format floats for better readability

    if all_similarities_df_rows:
        df_sim = pd.DataFrame(all_similarities_df_rows, columns=cfg.clip.prompts, index=all_img_names)
        df_prob = pd.DataFrame(all_probabilities_df_rows, columns=cfg.clip.prompts, index=all_img_names)
        print("\n--- Overall CLIP Similarity Scores (Pre-Softmax, Scaled by 100) ---")
        print(df_sim)
        print(df_prob)
    else:
        print("No image batches were processed to generate a DataFrame.")

    
    
    df_sim.to_csv(f"saved/{exp_name}/results/clip_similarity_scores.csv")
    df_prob.to_csv(f"saved/{exp_name}/results/clip_probability_scores.csv")

if __name__ == "__main__":
    main()