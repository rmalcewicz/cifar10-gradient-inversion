import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(6 * 6 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.adaptive_avg_pool2d(x, (6, 6))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ConvNet(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.seq_conv = nn.Sequential(
            self.conv1,
            self.relu,
            self.flatten,
        )

        self.fc1 = nn.Linear(self.compute_shape(input_shape), 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, n_classes)

        self.seq_fc = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu, self.fc3)

    def forward(self, x):
        x = self.seq_conv(x)
        x = self.seq_fc(x)
        return x

    def compute_shape(self, input_shape):
        x = torch.randn(input_shape).unsqueeze(0)
        x = self.seq_conv(x)
        return x.shape[1]

    def predict_proba(self, x):
        return self.softmax(self.forward(x))
