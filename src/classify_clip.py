import torch


def calculate_prediction(class_a, class_b, similarity_probabilities, labels_tensor):
    cifar10_class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    mean_prob_tensor = torch.zeros(2, 10, device=similarity_probabilities.device)
    final_top_class = torch.zeros(2, dtype=torch.long, device=mean_prob_tensor.device)

    mask_label_0 = labels_tensor == 0
    mask_label_1 = labels_tensor == 1

    if mask_label_0.any():
        mean_prob_tensor[0] = similarity_probabilities[mask_label_0].mean(dim=0)
    else:
        mean_prob_tensor[0] += (
            0.10  # if no images with label 0 then assign same prob for all classes
        )

    if mask_label_1.any():
        mean_prob_tensor[1] = similarity_probabilities[mask_label_1].mean(dim=0)
    else:
        mean_prob_tensor[1] += (
            0.10  # if no images with label 1 then assign same prob for all classes
        )

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
    success = [0, 0]
    if class_a == final_selected_class_names[0]:
        success[0] = 1
    if class_b == final_selected_class_names[1]:
        success[1] = 1
    succeses = sum(success)

    # print(succeses)
    all_counts = torch.bincount(labels_tensor, minlength=2)
    n_examples_0 = all_counts[0].item()
    n_examples_1 = all_counts[1].item()
    majority_label = torch.argmax(all_counts).item()
    final_result_dict = {
        "successes": succeses,
        "success_0": success[0],
        "success_1": success[1],
        "majority_success": success[majority_label],
        "selected_class_0": final_selected_class_names[0],
        "selected_class_1": final_selected_class_names[1],
        "n_examples_0": n_examples_0,
        "n_examples_1": n_examples_1,
        "majority_class": majority_label
    }
    return final_result_dict
