import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from models import PointNetEmbedding


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0

class SimplifiedDIS:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    def __call__(self, depth_profiles: torch.Tensor):
        return self.sample(depth_profiles)

    def init(self, params):
        self.au, self.bu, self.ad, self.bd = params.to(self.device)

    def up(self, x):
        return (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(params)
        xs_p = torch.rand(nevents, device=self.device)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)

        xs_n = torch.rand(nevents, device=self.device)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

        return torch.stack([sigma_p, sigma_n], dim=-1)


def generate_data(num_samples, num_events, theta_dim=4, device=device):
    simulator = SimplifiedDIS(device)

    thetas = torch.rand((num_samples, theta_dim), dtype=torch.float32, device=device) * 10
    xs = torch.stack([simulator.sample(theta, num_events) for theta in thetas]).to(device)

    return thetas, xs


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


def train(model, thetas, xs, num_epochs=10, batch_size=32, lr=1e-3, num_workers=0):  # Set num_workers=0
    device = thetas.device
    xs_tensor = advanced_feature_engineering(xs)
    
    # Move to CPU for DataLoader if using num_workers > 0
    thetas_cpu = thetas.cpu()
    xs_tensor_cpu = xs_tensor.cpu()
    
    dataset = TensorDataset(thetas_cpu, xs_tensor_cpu)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for theta_batch, x_batch in dataloader:
            theta_batch, x_batch = theta_batch.to(device), x_batch.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                embeddings = model(x_batch)
                # Calculate distances within the current batch only
                dists = torch.cdist(embeddings, embeddings, p=2)
                
                # Calculate target distances for the corresponding thetas in this batch
                target_dists = torch.cdist(theta_batch, theta_batch, p=2)
                
                loss = F.mse_loss(dists, target_dists)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return model, loss_history


# Run training
num_samples = 200
num_events = 100000
thetas, xs = generate_data(num_samples, num_events, device=device)

xs_tensor_engineered = advanced_feature_engineering(xs)
input_dim = xs_tensor_engineered.shape[-1]

model = PointNetEmbedding(input_dim=input_dim, latent_dim=128).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model, loss_history = train(model, thetas, xs, num_epochs=200, batch_size=32, lr=1e-3)

torch.save(model.state_dict(), 'pointnet_embedding.pth')
np.save('loss_history.npy', np.array(loss_history))
