import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleCNN, ConvNet
from src.data import get_binary_cifar10

def capture_batch(device, class_a, class_b, capture_batch_idx=[3000], experiment_name="ex1", batch_size=1):
    model = ConvNet(input_shape=(3, 32, 32), n_classes=2).to(device)
    loader = get_binary_cifar10(class_a, class_b, batch_size=batch_size)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        if i in capture_batch_idx:
            grad = [p.grad.clone().detach() for p in model.parameters()]
            flat_grad = torch.cat([g.view(-1) for g in grad])
            print(flat_grad.size())
            os.makedirs(f"saved/{experiment_name}", exist_ok=True)
            torch.save(images.detach().cpu(), f"saved/{experiment_name}/batch_images.pt")
            torch.save(labels.detach().cpu(), f"saved/{experiment_name}/batch_labels.pt")
            torch.save(flat_grad.cpu(), f"saved/{experiment_name}/batch_gradient.pt")
            torch.save(model.state_dict(), f"saved/{experiment_name}/model_state.pt")
            print(f"Saved batch {i}")
    
        optimizer.step()