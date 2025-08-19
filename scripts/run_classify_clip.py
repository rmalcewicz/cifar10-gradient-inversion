import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json


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
    exp_name = cfg.experiment.name
    run_idx = cfg.data.repetition
    batch_size = cfg.data.batch_size

    class_a = cfg.data.class_a
    class_b = cfg.data.class_b

    output_path = cfg.paths.clip_results_dir
    recons_path = os.path.join(cfg.paths.reconstruction_results_dir, "recons")
    original_path = cfg.paths.batch_data_dir

    os.makedirs(output_path, exist_ok=True)

    cifar10_class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    for idx in batch_idx:
        images_batch_file = os.path.join(recons_path, f"batch_{idx}_images.pt")
        labels_batch_file = os.path.join(recons_path, f"batch_{idx}_labels.pt")

        if cfg.data.original:
            images_batch_file = os.path.join(original_path, f"batch_{idx}/batch_images.pt")

        if not os.path.exists(images_batch_file):
            print(f"Batch file not found: {images_batch_file}. Skipping this batch.")
            continue

        images_tensor_batch = torch.load(images_batch_file)
        labels_tensor = torch.load(labels_batch_file)

        pil_images = [transforms.ToPILImage()(img_tensor.cpu()) for img_tensor in images_tensor_batch]
        inputs = processor(text=list_of_prompts, images=pil_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            similarity_scores = outputs.logits_per_image

            similarity_probabilities = similarity_scores.softmax(dim=-1)

        mean_prob_tensor = torch.zeros(2, 10, device=similarity_probabilities.device)
        final_top_class = torch.zeros(2, dtype=torch.long, device=mean_prob_tensor.device)

        mask_label_0 = labels_tensor == 0
        mask_label_1 = labels_tensor == 1

        if mask_label_0.any():
            mean_prob_tensor[0] = similarity_probabilities[mask_label_0].mean(dim=0)
        else:
            mean_prob_tensor[0] += 0.10  # if no images with label 0 then assign same prob for all classes

        if mask_label_1.any():
            mean_prob_tensor[1] = similarity_probabilities[mask_label_1].mean(dim=0)
        else:
            mean_prob_tensor[1] += 0.10  # if no images with label 1 then assign same prob for all classes

        top2_values, top2_indices = mean_prob_tensor.topk(k=2, dim=1)

        if not mask_label_0.any():
            top2_indices[0] = torch.randint(0, 10, (2,))

        if not mask_label_1.any():
            top2_indices[1] = torch.randint(0, 10, (2,))

        top1_label0_idx = top2_indices[0, 0]
        top1_label1_idx = top2_indices[1, 0]

        # Check for conflict
        if top1_label0_idx == top1_label1_idx:
            # Determine which label has higher confidence for the shared top class
            prob_label0_for_shared_class = top2_values[0, 0]
            prob_label1_for_shared_class = top2_values[1, 0]

            if prob_label0_for_shared_class >= prob_label1_for_shared_class:
                # Label 0 keeps the shared class (or if probabilities are equal, label 0 gets priority)
                final_top_class[0] = top1_label0_idx
                # Label 1 gets its second best class
                final_top_class[1] = top2_indices[1, 1]  # This is the 2nd best for label 1
                # print(f"  {reconstructed_label_to_name_map[0]} (prob: {prob_label0_for_shared_class:.4f}) gets {cifar10_class_names[final_top_class[0]]}.")
                # print(f"  {reconstructed_label_to_name_map[1]} (prob: {top2_values[1, 1]:.4f}) gets its 2nd choice: {cifar10_class_names[final_top_class[1]]}.")
            else:
                # Label 1 keeps the shared class
                final_top_class[1] = top1_label1_idx
                # Label 0 gets its second best class
                final_top_class[0] = top2_indices[0, 1]  # This is the 2nd best for label 0
                # print(f"  {reconstructed_label_to_name_map[1]} (prob: {prob_label1_for_shared_class:.4f}) gets {cifar10_class_names[final_top_class[1]]}.")
                # print(f"  {reconstructed_label_to_name_map[0]} (prob: {top2_values[0, 1]:.4f}) gets its 2nd choice: {cifar10_class_names[final_top_class[0]]}.")

        else:
            # No conflict, both labels get their top choice
            final_top_class[0] = top1_label0_idx
            final_top_class[1] = top1_label1_idx

        # print("\nFinal selected classes (indices):", final_top_class)
        # To get the names:
        final_selected_class_names = [cifar10_class_names[idx.item()] for idx in final_top_class]
        # print("Final selected class names:", final_selected_class_names)
        succeses = 0
        if class_a == final_selected_class_names[0]:
            succeses += 1
        if class_b == final_selected_class_names[1]:
            succeses += 1

        # print(succeses)
        final_result_dict = {
            "successes": succeses,
            "selected_class_0": final_selected_class_names[0],
            "selected_class_1": final_selected_class_names[1],
        }

        os.makedirs(output_path, exist_ok=True)
        json_path = output_path + f"/batch_{idx}_recon.json"
        if cfg.data.original:
            json_path = output_path + f"/batch_{idx}_original.json"
        with open(json_path, "w") as f:
            json.dump(final_result_dict, f, indent=4)
        # print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
