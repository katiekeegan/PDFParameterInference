import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from models import *
import matplotlib.pyplot as plt
torch.cuda.empty_cache()  # After each step

# Loss Function
def nll_loss(predicted_mean, predicted_var, true_params):
    """Negative log-likelihood loss."""
    loss = 0.5 * torch.mean(torch.log(predicted_var+1e-8) + (true_params - predicted_mean)**2 / predicted_var)
    return loss

# Feature Engineering
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

# SimplifiedDIS class (Simulator)
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
        return self.sample(depth_profiles)

    def init(self, params):
        self.Nu = 1
        self.au = params[0]
        self.bu = params[1]
        self.Nd = 2
        self.ad = params[2]
        self.bd = params[3]

    def up(self, x):
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(torch.tensor(params, device=self.device))

        xs_p = torch.rand(nevents, device=self.device)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)  # Replace NaNs with 0

        xs_n = torch.rand(nevents, device=self.device)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()
# Data Generation with improved stability
def generate_data(num_samples, num_events, theta_dim=4, x_dim=2, device=torch.device("cpu")):
    simulator = SimplifiedDIS(device)
    # ranges = [(-0.5, 10), (0.1, 10), (0.1, 10), (0.1, 10)]  # Avoid zero values
    # theta_bounds = torch.tensor([
    #             [0.1, 5],
    #             [-1, -0.1],
    #             [0.1, 5],
    #             [-1, -0.1],
    #         ], device=device)

    # # Sample uniform values in [0, 1] for each dimension
    # theta = torch.rand(theta_dim, device=device)
                # [0.1, 5],
                # [-1, -0.5],
                # [0.1, 5],
                # [-1, -0.5],
    # # Scale to the desired ranges
    # theta = theta * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]
    ranges = [(0.1, 5), (-1, -0.5), (0.1, 5), (-1, -0.5)]  # Example ranges
    thetas = np.column_stack([np.random.uniform(low, high, size=num_samples) for low, high in ranges])
    xs = np.array([simulator.sample(theta, num_events).cpu().numpy() for theta in thetas]) + 1e-8
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32, device=device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    return thetas_tensor, xs_tensor

class EventDataset(Dataset):
    def __init__(self, event_data, param_data, pointnet_model, device, batch_size=8):
        self.param_data = param_data
        self.device = device

        # Wrap event_data in a lightweight dataset for batching
        event_dataset = TensorDataset(event_data)
        event_loader = DataLoader(event_dataset, batch_size=batch_size)

        self.pointnet_model = pointnet_model.to(device)
        self.pointnet_model.eval()

        all_latents = []

        with torch.no_grad():
            for (batch,) in event_loader:
                batch = batch.to(device)
                latents = self.pointnet_model(batch).cpu()
                all_latents.append(latents)
                torch.cuda.empty_cache()  # After each step
        self.latent_embeddings = torch.cat(all_latents, dim=0)

    def __len__(self):
        return len(self.param_data)

    def __getitem__(self, idx):
        latent_embedding = self.latent_embeddings[idx]
        params = self.param_data[idx]
        return latent_embedding, params
# Mixture Density Network (MDN)
class MDN(nn.Module):
    def __init__(self, latent_dim, param_dim, num_components=3):
        super(MDN, self).__init__()
        self.param_dim = param_dim
        self.num_components = num_components
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.pi = nn.Linear(128, param_dim * num_components)
        self.mu = nn.Linear(128, param_dim * num_components)
        self.log_var = nn.Linear(128, param_dim * num_components)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        batch_size = x.size(0)
        pi = self.pi(x).view(batch_size, self.param_dim, self.num_components)
        pi = F.softmax(pi, dim=-1)
        mu = 10 * torch.sigmoid(self.mu(x).view(batch_size, self.param_dim, self.num_components))
        log_var = self.log_var(x).view(batch_size, self.param_dim, self.num_components)
        sigma = torch.exp(log_var)
        return pi, mu, sigma

def mdn_loss(log_pi, mu, cov, target):
    """
    log_pi: [B, K] (already in log domain from network)
    mu: [B, K, D]
    cov: [B, K, D, D]
    target: [B, D]
    """
    B, K, D = mu.shape
    target = target.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
    
    # Vectorized log prob (all components)
    try:
        dist = MultivariateNormal(
            loc=mu.reshape(B*K, D),
            scale_tril=cov.reshape(B*K, D, D)
        )
        log_probs = dist.log_prob(target.reshape(B*K, D)).view(B, K)
        
        # Stabilized computation
        log_pi = torch.log(log_pi + 1e-8)  # Already in log domain from network
        loss = -torch.logsumexp(log_pi + log_probs, dim=1).mean()
        
    except Exception as e:
        print(f"MDN error: {e}")
        print(f"Shapes: pi {log_pi.shape}, mu {mu.shape}, cov {cov.shape}, target {target.shape}")
        loss = torch.tensor(float('nan'), device=target.device)
        
    return loss

def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


def load_data(xs_file, thetas_file):
    with open(xs_file, "rb") as f:
        xs = pickle.load(f)
    with open(thetas_file, "rb") as f:
        thetas = pickle.load(f)
    return xs, thetas


def prepare_tensors(xs, thetas, device):
    xs_tensor = torch.tensor(xs, dtype=torch.float32).to(device)
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32).to(device)
    return xs_tensor, thetas_tensor


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (latent_embeddings, true_params) in enumerate(dataloader):
        latent_embeddings = latent_embeddings.to(device)
        true_params = true_params.to(device).squeeze()

        optimizer.zero_grad()
        neg_log_likelihood = -model(latent_embeddings, true_params).mean()
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += neg_log_likelihood.item()

    return total_loss / len(dataloader) 


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 2048
    model = ConditionalRealNVP(latent_dim=latent_dim, param_dim=4, hidden_dim=1024, num_flows=6)
    num_samples = 2000
    num_events = 500000
    thetas, xs = generate_data(num_samples, num_events)
    xs_tensor_engineered = advanced_feature_engineering(xs)
    input_dim = xs_tensor_engineered.shape[-1]
    pointnet_model = PointNetEmbedding(input_dim=input_dim, latent_dim=latent_dim)
    # Load the state dictionary from the file
    state_dict = torch.load('pointnet_embedding_latent_dim_2048.pth')

    # Remove the 'module.' prefix from the state_dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load the modified state_dict into the model
    pointnet_model.load_state_dict(new_state_dict)
    pointnet_model.eval()
    xs_tensor = advanced_feature_engineering(xs)
    dataset = EventDataset(xs_tensor, thetas, pointnet_model, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    del pointnet_model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = get_optimizer(model, lr=1e-3)
    scheduler = get_scheduler(optimizer)
    epochs = 10000
    # prune_components(model)
    for epoch in range(0, epochs):
        epoch_loss = train(model, dataloader, optimizer, device)
        print(f'Epoch: {epoch}, Loss: {epoch_loss}')
        if epoch % 100 == 0:
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "trained_conditional_normalizing_flow_model_latent_dim_1024_hidden_dim_1024.pth")
    print("Training complete and model saved.")


if __name__ == "__main__":
    main()
