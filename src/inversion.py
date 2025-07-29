import os
import torch
import torchvision
import torch.nn as nn
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR

from src.utils import TVLoss, Batch_data
from src.model import ConvNet
from src.evaluation import evaluate_reconstruction



def reconstruct(
    device: str,
    batch_data: Batch_data,
    output_path: str,
    batch_idx:int,
    num_iterations: int = 500,
    tv_coeff: float = 1e-6,
    lr: float = 0.1,
    wandb_log: bool = False 
):
    if wandb_log:
        import wandb

    
    true_grad = batch_data.gradient
    true_labels = batch_data.labels
    true_images = batch_data.images


    model = ConvNet(input_shape=(3, 32, 32), n_classes=2).to(device)
    model.load_state_dict(batch_data.model_state)
    model.eval()

    dummy_input = torch.randn((len(true_labels), 3, 32, 32)).to(device).detach()
    dummy_input.requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_input], lr=lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    cosine_similarity = nn.CosineSimilarity(dim=0)
    tv = TVLoss

    os.makedirs(f"{output_path}/batch_{batch_idx}", exist_ok=True)
    vutils.save_image(true_images.detach().cpu(), f"{output_path}/batch_{batch_idx}/true.png", nrow=8, normalize=True)
    if wandb_log:
        grid_image = vutils.make_grid(true_images.detach().cpu(), nrow=4, padding=2, normalize=True)
        image_for_wandb = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to(torch.uint8).numpy()
        wandb.log({"original_images_grid": wandb.Image(image_for_wandb)})
    
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
            vutils.save_image(dummy_input.detach().cpu(), f"{output_path}/batch_{batch_idx}/debug_{i}.png", nrow=8, normalize=True)
        
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
    os.makedirs(f"{output_path}/recons", exist_ok=True)
    torch.save(dummy_input.detach().cpu(), f"{output_path}/recons/batch_{batch_idx}.pt")
    vutils.save_image(dummy_input.detach().cpu(), f"{output_path}/batch_{batch_idx}/reconstruction.png", nrow=8, normalize=True)
    if wandb_log:
        grid_image = vutils.make_grid(dummy_input.detach().cpu(), nrow=4, padding=2, normalize=True)
        image_for_wandb = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to(torch.uint8).numpy()
        wandb.log({"reconstructed_images_grid": wandb.Image(image_for_wandb)})
    
    # Final evaluation
    metrics = evaluate_reconstruction(dummy_input.detach(), true_images)
    print(f"PSNR: {metrics['psnr']:.3f}, SSIM: {metrics['ssim']:.3f}")

    if wandb_log:
        wandb.log({
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"]
        })


    return dummy_input.detach().cpu()
