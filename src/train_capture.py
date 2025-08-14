import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleCNN, ConvNet
from src.data import get_binary_cifar10

# saves the batches information in /saved/exp_name each batch in individual dir batch_{i}
def capture_batch(
    device: str,
    class_a: str,
    class_b: str,
    output_path: str,
    capture_batch_idx: list = [100],
    experiment_name: str = "exp1",
    batch_size: int = 1,
    only_first: bool = True
):
    """
    Captures batch images, labels, gradients, and model states at specified batch indices.

    Args:
        device (str): Device to run the training on ('cuda' or 'cpu').
        class_a (int): The first class for binary CIFAR-10.
        class_b (int): The second class for binary CIFAR-10.
        base_save_dir (str): Path to the base directory for saving.
        capture_batch_idx (list, optional): List of batch indices at which to capture data.
                                             Defaults to [100].
        experiment_name (str, optional): Name of the experiment. Defaults to "ex1".
        batch_size (int, optional): Size of each training batch. Defaults to 1.
    """

    if not capture_batch_idx:
        print("capture_batch_idx is empty. No batches will be captured.")
        return
    
    model = ConvNet(input_shape=(3, 32, 32), n_classes=2).to(device)

    loader = get_binary_cifar10(class_a, class_b, batch_size=batch_size)
    if not loader:
        print("DataLoader is empty. Cannot perform training/capture.")
        return

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        if i in capture_batch_idx:
            if only_first:
                grad = [p.grad.clone().detach() for p in model[0].parameters()]
            else:
                grad = [p.grad.clone().detach() for p in model.parameters()]
                
            flat_grad = torch.cat([g.view(-1) for g in grad])

            batch_data_path = os.path.join(output_path, f"batch_{i}")

            os.makedirs(f"{batch_data_path}", exist_ok=True)
            torch.save(images.detach().cpu(), f"{batch_data_path}/batch_images.pt")
            torch.save(labels.detach().cpu(), f"{batch_data_path}/batch_labels.pt")
            torch.save(flat_grad.cpu(), f"{batch_data_path}/batch_gradient.pt")
            torch.save(model.state_dict(), f"{batch_data_path}/model_state.pt")

            #print(f"Saved batch {i}")

            if i >= max(capture_batch_idx):
                break
    
        optimizer.step()