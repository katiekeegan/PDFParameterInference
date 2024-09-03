import torch
import torch.nn as nn
from torch import cat
import torch.nn.functional as F


class ConvolutionalGAN(nn.Module):
    def __init__(self, noise_dim, param_dims):
        super().__init__()
        # Encoder
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 2 * noise_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(2 * noise_dim, 2 * noise_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(2 * noise_dim, int(param_dims)),
        )

    def forward(self, x):
        x1 = x
        x2 = self.model(x1)
        return x2


class SurrogatePhysicsModel(nn.Module):
    def __init__(self, input_dim):
        super(SurrogatePhysicsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # Apply Sigmoid activation function
        return x


class MLP(nn.Module):
    def __init__(self, input_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # First fully connected layer
        self.fc2 = nn.Linear(32, 16)  # Second fully connected layer
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.float()
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))  # Apply ReLU activation function
        x = self.fc3(x)
        x = self.sigmoid(x)  # Apply Sigmoid activation function
        return x
