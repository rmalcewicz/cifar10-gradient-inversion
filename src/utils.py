import torch
import os
import torchvision.transforms as T

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


def load_target_data(exp_dir, device):
    """
    Loads original target data saved during capture
    Returns (label_tensor, image tensor)
    """

    target_path = f"{exp_dir}/target.pt"
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"No target.pt in {exp_dir}")

    target = torch.load(target_path, map_location=device)
    label = target["label"]
    image = target["image"]
    return label, image


def TVLoss(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


class Batch_data:
    def __init__(self):
        self.gradient = None
        self.images = None
        self.labels = None
        self.model_state = None

    def load(self, batch_path, batch_n, skip=True):
        target_path = f"{batch_path}/batch_{batch_n}"
        if not skip:
            self.gradient = torch.load(f"{target_path}/batch_gradient.pt")
            self.model_state = torch.load(f"{target_path}/model_state.pt")
        self.images = torch.load(f"{target_path}/batch_images.pt")
        self.labels = torch.load(f"{target_path}/batch_labels.pt")

    def remove(self, batch_path, batch_n):
        target_path = f"{batch_path}/batch_{batch_n}"
        os.remove(f"{target_path}/batch_gradient.pt")
        os.remove(f"{target_path}/model_state.pt")


def grad_loss_fn(true_grad, grad, local=False):
    indices = torch.arange(len(true_grad))
    weights = true_grad[0].new_ones(len(true_grad))

    pnorm = [0, 0]
    costs = 0
    for i in indices:
        if local:
            costs += (
                1
                - torch.nn.functional.cosine_similarity(
                    grad[i].flatten(), true_grad[i].flatten(), 0, 1e-10
                )
                * weights[i]
            )
        else:
            costs -= (grad[i] * true_grad[i]).sum() * weights[i]
            pnorm[0] += grad[i].pow(2).sum() * weights[i]
            pnorm[1] += true_grad[i].pow(2).sum() * weights[i]

    if not local:
        costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

    return costs


clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]

clip_preprocess = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=clip_mean, std=clip_std),
    ]
)


def clip_guidance_loss(guidance, images, labels, model, processor, device):
    guidance = torch.tensor(guidance).to(device)

    guidance_classes = guidance[labels]
    guidance_mask = guidance_classes != -1

    if not guidance_mask.any():
        return torch.tensor(0.0, device=device)

    images_to_guide = images[guidance_mask]
    classes_to_guide = guidance_classes[guidance_mask]

    prompts = [
        f"a photo of a {cifar10_class_names[c]}" for c in classes_to_guide.cpu().numpy()
    ]

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    image_inputs = clip_preprocess(images_to_guide)
    image_inputs = image_inputs.to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    image_features = model.get_image_features(pixel_values=image_inputs)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    cosine_similarity = torch.sum(text_features * image_features, dim=-1)

    guidance_loss = (1 - cosine_similarity).mean()
    return guidance_loss
