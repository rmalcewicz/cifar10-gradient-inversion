import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR

from src.utils import load_gradients, TVLoss
from src.model import ConvNet
import torchvision

def reconstruct(
    grad_path: str,
    device: str,
    num_iterations: int = 500,
    tv_coeff: float = 1e-6,
    lr: float = 0.1,
    wandb_log: bool = False
):

    true_grad = torch.load(f"{grad_path}/batch_gradient.pt").to(device)
    true_labels = torch.load(f"{grad_path}/batch_labels.pt").to(device)
    true_images = torch.load(f"{grad_path}/batch_images.pt")


    model = ConvNet(input_shape=(3, 32, 32), n_classes=2).to(device)
    model.load_state_dict(torch.load(f"{grad_path}/model_state.pt"))
    model.eval()

    dummy_input = torch.randn((len(true_labels), 3, 32, 32)).to(device).detach()
    dummy_input.requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_input], lr=lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    cosine_similarity = nn.CosineSimilarity(dim=0)
    tv = TVLoss()

    os.makedirs(f"{grad_path}/results", exist_ok=True)
    vutils.save_image(true_images.detach().cpu(), f"{grad_path}/results/true.png", nrow=8, normalize=True)
    
    for i in range(num_iterations):
        optimizer.zero_grad()

        output = model(dummy_input)
        loss = criterion(output, true_labels)

        dummy_grad_1 = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        dummy_grad = torch.cat([g.view(-1) for g in dummy_grad_1])

        cos_sim = cosine_similarity(dummy_grad, true_grad)
        grad_loss = (1 - cos_sim).mean()

        tv_loss = tv(dummy_input)
        total_loss = grad_loss + tv_coeff * tv_loss       

        total_loss.backward()
        optimizer.step()
        dummy_input.data.clamp_(0, 1)
        scheduler.step()

        # Save debug image every 200 iters
        if i % 200 == 0:
            vutils.save_image(dummy_input.detach().cpu(), f"{grad_path}/results/debug_{i}.png", nrow=8, normalize=True)
        
        # Log to wandb
            if wandb_log and (i % 10 == 0 or i == num_iterations - 1):
                import wandb
                wandb.log({
                    "iteration": i,
                    "grad_loss": grad_loss.item(),
                    "tv_loss": tv_loss.item(),
                    "total_loss": total_loss.item()
                })
        
        # Print to console
        if i % 50 == 0:
            print(f"[{i}] Gradient loss: {grad_loss.item():.10f}, Total loss: {total_loss.item():.10f}")
        
        # Sanity check
        assert dummy_grad.shape == true_grad.shape, f"{dummy_grad.shape} vs {true_grad.shape}"

    # Final output
    vutils.save_image(dummy_input.detach().cpu(), f"{grad_path}/results/reconstruction.png", nrow=8, normalize=True)
    if wandb_log:
        wandb.log({"reconstruction": wandb.Image(dummy_input.detach().cpu())})
    
    return dummy_input.detach().cpu()
