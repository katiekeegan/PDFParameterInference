import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import csv

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimplifiedDIS:
    def __init__(self):
        self.Nu = 1
        self.au = 1  # params[0]
        self.bu = 1  # params[1]
        self.Nd = 2
        self.ad = 1  # params[2]
        self.bd = 1  # params[3]

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
        self.init(params)

        xs_p = torch.rand(nevents)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)  # Replace NaNs with 0

        xs_n = torch.rand(nevents)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()


# Generate synthetic dataset
def generate_data(num_samples, num_events, theta_dim=4, x_dim=2, device=torch.device("cpu")):
    """
    Generate a dataset of (theta, x) pairs using the simulator.
    """
    simulator = SimplifiedDIS()
    thetas = np.random.uniform(low=0, high=5.0, size=(num_samples, theta_dim))
    xs = np.array([simulator.sample(theta, num_events).numpy() for theta in thetas])
    return thetas, xs

# RealNVP Coupling Layer
class RealNVPCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2

        # Neural network to predict scale and shift
        self.scale_shift_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.half_dim) * 2)
        )

    def forward(self, x, reverse=False):
        """
        Forward or inverse transformation for the coupling layer.
        """
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]

        # Predict scale and shift
        scale_shift = self.scale_shift_net(x1)
        scale = torch.sigmoid(scale_shift[:, :(self.dim - self.half_dim)] + 2.0)
        shift = scale_shift[:, (self.dim - self.half_dim):]

        if not reverse:
            # Forward transformation
            y2 = x2 * scale + shift
            y1 = x1
            log_det = torch.sum(torch.log(scale), dim=-1)
        else:
            # Inverse transformation
            y2 = (x2 - shift) / scale
            y1 = x1
            log_det = -torch.sum(torch.log(scale), dim=-1)

        y = torch.cat([y1, y2], dim=-1)
        return y, log_det

# RealNVP Flow
class RealNVPFlow(nn.Module):
    def __init__(self, dim, num_layers=4, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # Stack of coupling layers
        self.layers = nn.ModuleList([
            RealNVPCouplingLayer(dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, reverse=False):
        """
        Forward or inverse transformation for the flow.
        """
        log_det = 0.0
        if not reverse:
            for layer in self.layers:
                x, ld = layer(x, reverse=False)
                log_det += ld
        else:
            for layer in reversed(self.layers):
                x, ld = layer(x, reverse=True)
                log_det += ld
        return x, log_det

# Neural network to parameterize the flow
class FlowPosteriorNetwork(nn.Module):
    def __init__(self, x_dim, theta_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        self.theta_dim = theta_dim

        # Network to predict flow parameters
        self.flow_params_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers * theta_dim * 2)  # Parameters for flow layers
        )

        # Base distribution (e.g., standard Gaussian)
        self.base_dist = torch.distributions.Normal(torch.zeros(theta_dim).to(device),
                                                   torch.ones(theta_dim).to(device))

        # RealNVP flow
        self.flow = RealNVPFlow(theta_dim, num_layers, hidden_dim)

    def forward(self, x):
        """
        Predict the parameters of the flow conditioned on x.
        """
        flow_params = self.flow_params_net(x)
        return flow_params

    def log_prob(self, theta, x):
        """
        Compute the log probability of theta under the flow-based posterior.
        """
        # Transform theta to base distribution using the flow
        z, log_det = self.flow(theta, reverse=True)
        # Compute log probability under the base distribution
        log_prob = self.base_dist.log_prob(z).sum(dim=-1) + log_det
        return log_prob
# Training loop
def train_flow_posterior(model, dataloader, optimizer, epochs=1000, save_path="model.pth", loss_log_path="loss_log.csv"):
    model.train()
    loss_values = []  # List to store loss values

    for epoch in range(epochs):
        epoch_loss = 0.0
        for theta, x in dataloader:
            # Move data to device
            theta = theta.to(device)
            x = x.to(device)

            # Compute log probability of theta under the flow-based posterior
            log_prob = model.log_prob(theta, x)
            loss = -log_prob.mean()  # Negative log-likelihood loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss}")

    # Save the model's state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save the loss values to a CSV file
    with open(loss_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])
        for epoch, loss in enumerate(loss_values):
            writer.writerow([epoch, loss])
    print(f"Loss values saved to {loss_log_path}")

# Main
if __name__ == "__main__":
    # Hyperparameters
    theta_dim = 4
    x_dim = 2
    num_samples = 100  # Number of parameter settings
    num_events = 10000    # Number of events per parameter setting
    batch_size = 1024
    hidden_dim = 64
    lr = 1e-3
    epochs = 1000
    num_layers=4
    multi_log = True
    log = False

    # Generate synthetic data
    thetas, xs = generate_data(num_samples, num_events, theta_dim, x_dim, device)

    # Create dataset with pairs of (theta, single_event)
    thetas_repeated = np.repeat(thetas, num_events, axis=0)
    thetas_repeated = torch.tensor(thetas_repeated, dtype=torch.float32).to(device)
    xs_flattened = xs.reshape(-1, x_dim)  # Flatten events into (num_samples * num_events, x_dim)
    xs_flattened = torch.tensor(xs_flattened, dtype=torch.float32).to(device)
    if multi_log:
        xs_flattened = torch.cat([torch.log(xs_flattened + 1e-8), torch.log10(xs_flattened + 1e-8)], dim=-1).float()
    elif log:
        xs_flattened = torch.log(xs_flattened + 1e-8)

    # Convert to PyTorch tensors
    dataset = TensorDataset(thetas_repeated, xs_flattened)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    if multi_log:
        model = FlowPosteriorNetwork(x_dim*2, theta_dim, hidden_dim, num_layers).to(device)
    elif log:
        model = FlowPosteriorNetwork(x_dim, theta_dim, hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_flow_posterior(model, dataloader, optimizer, epochs, save_path="flow_posterior_model.pth", loss_log_path="loss_log.csv")