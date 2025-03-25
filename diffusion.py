import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimplifiedDIS:
    def __init__(self, device=None):
        self.Nu = 1
        self.au = 1  # params[0]
        self.bu = 1  # params[1]
        self.Nd = 2
        self.ad = 1  # params[2]
        self.bd = 1  # params[3]
        self.device = device

    def __call__(self, depth_profiles: np.ndarray):
        """Call to simulator."""
        return self.sample(depth_profiles)

    def init(self, params):
        self.Nu = 1
        self.au = params[0]
        self.bu = params[1]
        self.Nd = 2
        self.ad = params[2]
        self.bd = params[3]

    def up(self, x):
        u = self.Nu * (x ** self.au) * ((1 - x) ** self.bu)
        return u

    def down(self, x):
        d = self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)
        return d

    def sample(self, params, nevents=1):
        self.init(torch.tensor(params, device=self.device))

        xs_p = torch.rand(nevents, device=self.device)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)  # Replace NaNs with 0

        xs_n = torch.rand(nevents, device=self.device)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()


# Generate synthetic dataset
def generate_data(num_samples, num_events, theta_dim=4, x_dim=2, device=torch.device("cpu")):
    """
    Generate a dataset of (theta, x) pairs using the simulator.
    """
    simulator = SimplifiedDIS(device)
    thetas = np.random.uniform(low=0, high=5.0, size=(num_samples, theta_dim))
    xs = np.array([simulator.sample(theta, num_events).cpu().numpy() for theta in thetas])
    return thetas, xs


# Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, theta_dim, x_dim, hidden_dim=128):
        super().__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim

        # Network to predict noise
        self.net = nn.Sequential(
            nn.Linear(theta_dim + x_dim + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, theta_dim)
        )

    def forward(self, theta_t, t, x):
        """
        Forward pass of the model.
        theta_t: Noisy parameters at timestep t.
        t: Timestep.
        x: Observed data.
        """
        # Flatten x (batch_size, num_events, x_dim) -> (batch_size, num_events * x_dim)
        x = x.view(x.shape[0], -1)
        # Concatenate theta_t, x, and timestep t
        t = t.float() / 100.0  # Normalize timestep
        model_input = torch.cat([theta_t, x, t.unsqueeze(-1)], dim=-1)
        # Predict noise
        return self.net(model_input)


# Diffusion process
class DiffusionProcess:
    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=50):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, theta_0, t):
        """
        Forward process: Add noise to theta_0 at timestep t.
        """
        noise = torch.randn_like(theta_0).to(device)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        theta_t = torch.sqrt(alpha_bar_t) * theta_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return theta_t, noise


# Training loop
def train_diffusion(model, diffusion, dataloader, optimizer, epochs=1000):
    model.train()
    for epoch in range(epochs):
        for theta_0, x in dataloader:
            # Move data to the correct device
            theta_0 = theta_0.to(device)
            x = x.to(device)

            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (theta_0.shape[0],)).to(device)
            # Forward process: Add noise to theta_0
            theta_t, noise = diffusion.forward(theta_0, t)
            # Predict noise
            predicted_noise = model(theta_t, t, x)
            # Compute loss
            loss = nn.functional.mse_loss(predicted_noise, noise)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Main
if __name__ == "__main__":
    # Hyperparameters
    theta_dim = 4
    x_dim = 2
    num_samples = 100
    num_events = 5000
    batch_size = 64
    hidden_dim = 128
    lr = 1e-3
    epochs = 1000
    multi_log = True
    log = False

    # Generate synthetic data
    thetas, xs = generate_data(num_samples, num_events, theta_dim, x_dim, device)

    # Create dataset with pairs of (theta, single_event)
    # Repeat each theta num_events times and flatten the events
    thetas_repeated = np.repeat(thetas, num_events, axis=0)
    thetas_repeated = torch.tensor(thetas_repeated, dtype=torch.float32).to(device)
    xs_flattened = xs.reshape(-1, x_dim)  # Flatten events into (num_samples * num_events, x_dim)
    xs_flattened = torch.tensor(xs_flattened, dtype=torch.float32).to(device)
    if multi_log:
        xs_flattened = torch.cat([torch.log(xs_flattened + 1e-8), torch.log10(xs_flattened + 1e-8)], dim=-1).float()
    if log:
        xs_flattened = torch.log(xs_flattened + 1e-8)
    # Convert to PyTorch tensors
    dataset = TensorDataset(thetas_repeated, xs_flattened)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, diffusion process, and optimizer
    if multi_log:
        model = DiffusionModel(theta_dim, 2 * x_dim, hidden_dim).to(device)
    else:
        model = DiffusionModel(theta_dim, x_dim, hidden_dim).to(device)
    diffusion = DiffusionProcess()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_diffusion(model, diffusion, dataloader, optimizer, epochs)