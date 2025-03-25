import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Multivariate Gaussian Posterior
class GaussianPosterior(nn.Module):
    def __init__(self, dim=4):
        super(GaussianPosterior, self).__init__()
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(dim))  # Learnable mean vector
        self.log_std = nn.Parameter(torch.zeros(dim))  # Learnable log std (for numerical stability)

    def forward(self):
        std = torch.exp(self.log_std)  # Convert log std to std
        return dist.MultivariateNormal(self.mean, torch.diag(std**2))  # Diagonal covariance matrix

# Data Generation Function (updated to match dimensions)
def generate_data(num_samples, num_events, theta_dim=4, x_dim=4, device=torch.device("cpu")):
    ranges = [(-10, -5), (0.1, 5), (-2, -1), (7, 9)]
    thetas = np.column_stack([np.random.uniform(low, high, size=num_samples) for low, high in ranges])
    xs = np.random.randn(num_samples, num_events, x_dim)  # Now x_dim = 4 for consistency

    # Repeat theta values for all events
    thetas_repeated = np.repeat(thetas, num_events, axis=0)
    xs_flattened = xs.reshape(-1, x_dim)  # Flatten events into (num_samples * num_events, x_dim)

    # Convert to PyTorch tensors
    thetas_repeated = torch.tensor(thetas_repeated, dtype=torch.float32)
    xs_flattened = torch.tensor(xs_flattened, dtype=torch.float32)

    # Create dataset
    dataset = TensorDataset(thetas_repeated.to(device), xs_flattened.to(device))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    return dataloader

# Training Setup
def train_gaussian_posterior_model(dataloader, model, optimizer, num_epochs=1000):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for thetas, xs in dataloader:
            optimizer.zero_grad()

            # Forward pass through the model (multivariate Gaussian posterior)
            posterior = model()
            log_prob = posterior.log_prob(xs).sum()  # Log-likelihood

            # Backpropagation
            loss = -log_prob  # We minimize negative log-likelihood
            print(loss)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

# Initialize Model and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaussianPosterior(dim=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate synthetic data
num_samples = 1000
num_events = 10000
dataloader = generate_data(num_samples, num_events, device=device)

# Train the Gaussian posterior model
train_gaussian_posterior_model(dataloader, model, optimizer)
