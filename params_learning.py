import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models import *

def nll_loss(predicted_mean, predicted_var, true_params):
    """Negative log-likelihood loss."""
    loss = 0.5 * torch.mean(torch.log(predicted_var+1e-8) + (true_params - predicted_mean)**2 / predicted_var)
    return loss

def advanced_feature_engineering(xs_tensor):
    log_features = torch.log1p(xs_tensor)
    symlog_features = torch.sign(xs_tensor) * torch.log1p(xs_tensor.abs())

    ratio_features = []
    diff_features = []
    for i in range(xs_tensor.shape[-1]):
        for j in range(i + 1, xs_tensor.shape[-1]):
            ratio = xs_tensor[..., i] / (xs_tensor[..., j] + 1e-8)
            ratio_features.append(torch.log1p(ratio.abs()).unsqueeze(-1))

            diff = torch.log1p(xs_tensor[..., i]) - torch.log1p(xs_tensor[..., j])
            diff_features.append(diff.unsqueeze(-1))

    ratio_features = torch.cat(ratio_features, dim=-1)
    diff_features = torch.cat(diff_features, dim=-1)

    return torch.cat([log_features, symlog_features, ratio_features, diff_features], dim=-1)


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


def generate_data(num_samples, num_events, theta_dim=4, x_dim=2, device=torch.device("cpu")):
    """
    Generate a dataset of (theta, x) pairs using the simulator.
    theta: (num_samples, theta_dim)
    x: (num_samples, num_events, x_dim)
    """
    simulator = SimplifiedDIS(device)
    
    # Define the parameter ranges for the thetas
    ranges = [(0, 10), (0,10), (0,10), (0,10)]  # Example ranges
    
    # Generate thetas within the defined ranges
    thetas = np.column_stack([np.random.uniform(low, high, size=num_samples) for low, high in ranges])
    
    # Generate xs based on the thetas using the simulator
    xs = np.array([simulator.sample(theta, num_events).cpu().numpy() for theta in thetas]) + 1e-8
    
    # Convert the generated numpy arrays to PyTorch tensors
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32, device=device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    
    return thetas_tensor, xs_tensor


class EventDataset(Dataset):
    def __init__(self, event_data, param_data, pointnet_model, device):
        self.event_data = event_data
        self.param_data = param_data
        self.pointnet_model = pointnet_model
        self.device = device

    def __len__(self):
        return len(self.event_data)

    def __getitem__(self, idx):
        events = self.event_data[idx].to(self.device)  # Move the events tensor to the correct device
        params = self.param_data[idx].to(self.device)  # Move the params tensor to the correct device
        
        # Get latent embedding from pretrained PointNet model
        latent_embedding = self.pointnet_model(events.unsqueeze(0).to(self.device))  # Ensure it is on the correct device
        return latent_embedding, params

class MDN(nn.Module):
    def __init__(self, latent_dim, param_dim, num_components=3):
        """
        latent_dim: Dimension of the latent representation.
        param_dim: Number of parameters to predict.
        num_components: Number of mixture components.
        """
        super(MDN, self).__init__()
        self.param_dim = param_dim
        self.num_components = num_components
        
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Each output now predicts (param_dim * num_components) values.
        self.pi = nn.Linear(64, param_dim * num_components)
        self.mu = nn.Linear(64, param_dim * num_components)
        self.log_var = nn.Linear(64, param_dim * num_components)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        batch_size = x.size(0)
        
        # Reshape to (batch_size, param_dim, num_components)
        pi = self.pi(x).view(batch_size, self.param_dim, self.num_components)
        pi = F.softmax(pi, dim=-1)
        
        mu = 10 * torch.sigmoid(self.mu(x).view(batch_size, self.param_dim, self.num_components))
        log_var = self.log_var(x).view(batch_size, self.param_dim, self.num_components)
        sigma = torch.exp(log_var)
        return pi, mu, sigma

def mdn_loss(pi, mu, sigma, target):
    """
    Computes the negative log-likelihood loss for an MDN.
    - pi: Mixture weights of shape (batch_size, param_dim, num_components)
    - mu: Means of shape (batch_size, param_dim, num_components)
    - sigma: Standard deviations of shape (batch_size, param_dim, num_components)
    - target: True parameters of shape (batch_size, param_dim)
    """
    # Expand target to have a new dimension for the components
    target = target.unsqueeze(2).expand_as(mu)  # Now shape: (batch_size, param_dim, num_components)
    
    # Compute log probability for each Gaussian component
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(target)  # (batch_size, param_dim, num_components)
    
    # Weight by the mixture weights and sum using LogSumExp for stability
    weighted_log_prob = torch.log(pi + 1e-8) + log_prob  # Avoid log(0)
    log_sum = torch.logsumexp(weighted_log_prob, dim=2)  # Sum over mixture components
    
    loss = -torch.mean(log_sum)  # Negative log-likelihood loss
    return loss



def train(model, pointnet_model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for latent_embeddings, true_params in dataloader:
        latent_embeddings = latent_embeddings.to(device)
        true_params = true_params.to(device)

        optimizer.zero_grad()

        # pred_mean, pred_log_var = model(latent_embeddings)
        # loss = nll_loss(pred_mean, pred_log_var, true_params)
        pi, mu, sigma = model(latent_embeddings)  # MDN outputs
        loss = mdn_loss(pi, mu, sigma, true_params.squeeze())  # Adjust true_params shape if needed

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def main():
    # Assuming latent_dim and param_dim are defined
    latent_dim = 128  # Example latent dimension

    # Instantiate the model
    # model = LatentToParamsNN(latent_dim, 4)
    model = MDN(latent_dim, 4)
    # Example usage:
    # Simulate some data for `xs`
    num_samples = 1000
    num_events = 100000
    thetas, xs = generate_data(num_samples, num_events)
    # multi_log=True
    # log=False
    # # Convert `xs` to torch tensor
    # xs_tensor = torch.tensor(xs, dtype=torch.float32)
    # if multi_log:
    #     xs_tensor = torch.cat([torch.log(xs_tensor + 1e-8), torch.log10(xs_tensor + 1e-8)], dim=-1).float()
    # elif log:
    #     xs_tensor = torch.log(xs_tensor + 1e-8)
    xs_tensor_engineered = advanced_feature_engineering(xs)
    input_dim = xs_tensor_engineered.shape[-1]
    # Example input: latent embeddings of shape (batch_size, latent_dim)
    # Create the model again
    pointnet_model = PointNetEmbedding(input_dim=input_dim, latent_dim=128)

    # Load the saved state_dict into the model
    pointnet_model.load_state_dict(torch.load('pointnet_embedding_1000.pth'))
    pointnet_model.eval()  

    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pointnet_model.to(device)

    # Create Dataset and DataLoader
    xs_tensor = advanced_feature_engineering(xs)
    dataset = EventDataset(xs_tensor, thetas, pointnet_model, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_epochs = 300  # Set the number of epochs
    for epoch in range(num_epochs):
        avg_loss = train(model, pointnet_model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'latent_to_params_model.pth')
    evaluate_and_plot(model, pointnet_model, dataloader, device)

import matplotlib.pyplot as plt

def predict_with_uncertainty(model, latent_embedding):
    model.eval()
    with torch.no_grad():
        mean, log_var = model(latent_embedding)
        std = torch.exp(0.5 * log_var)  # Convert log variance to standard deviation
        sampled_theta = mean + std * torch.randn_like(std)  # Sample from predicted distribution
    return sampled_theta, mean, std


def evaluate_and_plot(model, pointnet_model, dataloader, device):
    model.eval()
    true_params_list = []
    predicted_means = []
    predicted_stds = []

    with torch.no_grad():
        for latent_embeddings, true_params in dataloader:
            latent_embeddings = latent_embeddings.to(device)
            true_params = true_params.to(device)

            pred_mean, pred_log_var = model(latent_embeddings)
            pred_std = torch.exp(0.5 * pred_log_var)

            true_params_list.append(true_params.cpu().numpy())
            predicted_means.append(pred_mean.cpu().numpy())
            predicted_stds.append(pred_std.cpu().numpy())

    true_params_array = np.concatenate(true_params_list)
    pred_means_array = np.concatenate(predicted_means)
    pred_stds_array = np.concatenate(predicted_stds)

    # Plot the predicted mean with confidence intervals
    plt.figure(figsize=(10, 6))
    for i in range(true_params_array.shape[1]):  # Loop over each parameter
        plt.errorbar(
            range(len(true_params_array)), pred_means_array[:, i],
            yerr=2 * pred_stds_array[:, i], fmt='o', label=f'Parameter {i}'
        )
    plt.legend()
    plt.xlabel('Sample index')
    plt.ylabel('Predicted parameter values')
    plt.title('Predicted parameters with uncertainty')
    plt.savefig('Uncertainty.png')
if __name__ == "__main__":
    main()
    # After training, call the evaluate_and_plot function
