import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from src.model import SimpleCNN, ConvNet
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR

def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

def reconstruct(
    device,
    lr: float = 0.1,
    iterations: int = 500,
    tv_weight = 1e-4,
    grad_path: str = "saved/exp1"):

    true_grad = torch.load(f"{grad_path}/batch_gradient.pt").to(device)
    true_labels = torch.load(f"{grad_path}/batch_labels.pt").to(device)
    true_images = torch.load(f"{grad_path}/batch_images.pt")

    print(true_grad.dtype)

    model = ConvNet(input_shape=(3, 32, 32), n_classes=2).to(device)
    model.load_state_dict(torch.load(f"{grad_path}/model_state.pt"))
    model.eval()

    dummy_input = torch.randn((len(true_labels), 3, 32, 32)).to(device).detach()
    dummy_input.requires_grad_(True)
    
    optimizer = torch.optim.Adam([dummy_input], lr=lr)
    criterion = nn.CrossEntropyLoss()

    cosine_similarity = nn.CosineSimilarity(dim=0)

    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    for i in range(iterations):
        optimizer.zero_grad()

        output = model(dummy_input)
        loss = criterion(output, true_labels)

        dummy_grad_1 = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        dummy_grad = torch.cat([g.view(-1) for g in dummy_grad_1])

        cos_sim = cosine_similarity(dummy_grad, true_grad)
        grad_loss = (1 - cos_sim).mean()

        tv_loss = total_variation(dummy_input)
        model.zero_grad()

        total_loss = grad_loss + tv_weight * tv_loss

        total_loss.backward()
        optimizer.step()
        
        dummy_input.data.clamp_(0, 1) 

        scheduler.step()
        if i % 200 == 0:
            vutils.save_image(dummy_input.detach().cpu(), f"results/exp1/debug_{i}.png", nrow=8, normalize=True)
        if i % 50 == 0:
            print(f"[{i}] Gradient loss: {grad_loss.item():.10f}, Total loss: {total_loss.item():.10f}")
        assert dummy_grad.shape == true_grad.shape, f"{dummy_grad.shape} vs {true_grad.shape}"
        vutils.save_image(true_images.detach().cpu(), f"results/exp1/true.png", nrow=8, normalize=True)

    return dummy_input.detach().cpu()
