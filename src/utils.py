import torch
import os


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
                - torch.nn.functional.cosine_similarity(grad[i].flatten(), true_grad[i].flatten(), 0, 1e-10)
                * weights[i]
            )
        else:
            costs -= (grad[i] * true_grad[i]).sum() * weights[i]
            pnorm[0] += grad[i].pow(2).sum() * weights[i]
            pnorm[1] += true_grad[i].pow(2).sum() * weights[i]

    if not local:
        costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

    return costs
