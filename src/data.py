from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import torchvision

class RemappedSubset(Dataset):
    def __init__(self, subset, class_a_idx, class_b_idx):
        self.subset = subset
        self.mapping = {class_a_idx: 0, class_b_idx: 1}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, label = self.subset[idx]
        return x, self.mapping[label]

def get_binary_cifar10(class_a, class_b, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    class_to_idx = {c: i for i, c in enumerate(cifar.classes)}
    idx_a, idx_b = class_to_idx[class_a], class_to_idx[class_b]
    selected_indices = [i for i, (_, label) in enumerate(cifar) if label in [idx_a, idx_b]]

    binary_subset = Subset(cifar, selected_indices)
    remapped_dataset = RemappedSubset(binary_subset, idx_a, idx_b)

    return DataLoader(remapped_dataset, batch_size=batch_size, shuffle=True)
    #return DataLoader(cifar, batch_size=batch_size, shuffle=True)