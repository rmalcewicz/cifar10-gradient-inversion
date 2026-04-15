import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json

from src.classify_clip import calculate_prediction


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg.clip.model
    if model_name == "ViT-B/32":
        model_name = "openai/clip-vit-base-patch32"
    elif model_name == "ViT-L/14":
        model_name = "openai/clip-vit-large-patch14"

    # Download model fgfrom HF
    # model = CLIPModel.from_pretrained(model_name).eval().to(device)
    # processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

    model_path = cfg.paths.clip_base_path

    model = CLIPModel.from_pretrained(model_path).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)

    list_of_prompts = OmegaConf.to_container(cfg.clip.prompts, resolve=True)

    batch_idx = cfg.data.capture_batch_idx
    # exp_name = cfg.experiment.name
    # run_idx =cfg.data.repetition
    # batch_size = cfg.data.batch_size

    n_aggregate = cfg.data.n_aggregate

    class_a = cfg.data.class_a
    class_b = cfg.data.class_b

    output_path = cfg.paths.clip_results_dir
    recons_path = os.path.join(cfg.paths.reconstruction_results_dir, "recons")
    original_path = cfg.paths.batch_data_dir

    os.makedirs(output_path, exist_ok=True)

    all_labels = []
    all_sim = []
    agg_counter = 0
    for i, idx in enumerate(batch_idx):
        images_batch_file = os.path.join(recons_path, f"batch_{idx}_images.pt")
        labels_batch_file = os.path.join(recons_path, f"batch_{idx}_labels.pt")

        if cfg.data.original:
            images_batch_file = os.path.join(original_path, f"batch_{idx}/batch_images.pt")

        if not os.path.exists(images_batch_file):
            print(f"Batch file not found: {images_batch_file}. Skipping this batch.")
            continue

        images_tensor_batch = torch.load(images_batch_file)
        labels_tensor = torch.load(labels_batch_file)

        pil_images = [
            transforms.ToPILImage()(img_tensor.cpu()) for img_tensor in images_tensor_batch
        ]
        inputs = processor(
            text=list_of_prompts, images=pil_images, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            similarity_scores = outputs.logits_per_image

            similarity_probabilities = similarity_scores.softmax(dim=-1)

        final_result_dict = calculate_prediction(
            class_a, class_b, similarity_probabilities, labels_tensor
        )

        all_labels.append(labels_tensor)
        all_sim.append(similarity_probabilities)

        os.makedirs(output_path, exist_ok=True)
        json_path = output_path + f"/batch_{idx}_recon.json"
        if cfg.data.original:
            json_path = output_path + f"/batch_{idx}_original.json"
        with open(json_path, "w") as f:
            json.dump(final_result_dict, f, indent=4)
        # print(f"Results saved to: {json_path}")

        if cfg.data.aggregate and ((i+1) % n_aggregate) == 0:
            aggregated_result_dict = calculate_prediction(
                class_a, class_b, torch.cat(all_sim), torch.cat(all_labels)
            )
            json_agg_suffix = "original" if cfg.data.original else "recon"
            json_agg_path = os.path.join(
                output_path, 
                f"{json_agg_suffix}_agg_{agg_counter}.json"
            )
            
            with open(json_agg_path, "w") as f:
                json.dump(aggregated_result_dict, f, indent=4)
            
            agg_counter += 1

            all_labels = []
            all_sim = []
            
if __name__ == "__main__":
    main()
