import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import PointNetEmbedding
import argparse

# Set up argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with simplified DIS simulation.")
    parser.add_argument('--num_samples', type=int, default=200, help="Number of samples to generate")
    parser.add_argument('--num_events', type=int, default=1000000, help="Number of events to simulate")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use for training")
    return parser.parse_args()

# Device setup
def setup_device(device_str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[1] if device_str.startswith("cuda") else ""
    return device

# Define the SimplifiedDIS class
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

# Data generation function
def generate_data(num_samples, num_events, theta_dim=4, device=torch.device("cpu")):
    simulator = SimplifiedDIS(device)
    thetas = torch.rand((num_samples, theta_dim), dtype=torch.float32, device=device) * 10
    xs = torch.stack([simulator.sample(theta, num_events) for theta in thetas]).to(device)
    return thetas, xs

# Feature engineering function
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

# Contrastive loss definition
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, scale=2.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, thetas):
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.mm(embeddings, embeddings.t())
        emb_dists = 1.0 - sim_matrix
        theta_dists = torch.cdist(thetas, thetas)
        theta_dists = theta_dists / (theta_dists.max() + 1e-8)

        sim_mask = (theta_dists < 0.15).float()
        dissim_mask = (theta_dists > 0.35).float()

        loss_sim = (sim_mask * emb_dists).mean()
        loss_dissim = (dissim_mask * F.relu(self.margin - emb_dists)).mean()

        return (loss_sim + self.scale * loss_dissim) / (1 + self.scale)

# Training function
def train(model, thetas, xs, num_epochs=10, batch_size=32, lr=1e-3, num_workers=0):
    device = thetas.device
    xs_tensor = advanced_feature_engineering(xs)

    thetas_cpu = thetas.cpu()
    xs_tensor_cpu = xs_tensor.cpu()

    dataset = TensorDataset(thetas_cpu, xs_tensor_cpu)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = ContrastiveLoss(margin=1.0)
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for theta_batch, x_batch in dataloader:
            theta_batch, x_batch = theta_batch.to(device), x_batch.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                embeddings = model(x_batch)
                l2_reg = 0.001 * torch.mean(torch.norm(embeddings, p=2, dim=1))
                loss = loss_fn(embeddings, theta_batch) + l2_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return model, loss_history

# Main function to run the training
def main():
    args = parse_args()
    device = setup_device(args.device)

    # Generate data
    thetas, xs = generate_data(args.num_samples, args.num_events, device=device)
    
    # Feature engineering
    xs_tensor_engineered = advanced_feature_engineering(xs)
    input_dim = xs_tensor_engineered.shape[-1]

    # Initialize model
    model = PointNetEmbedding(input_dim=input_dim, latent_dim=128).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Train the model
    model, loss_history = train(model, thetas, xs, num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr)

    # Save model and loss history
    torch.save(model.state_dict(), 'pointnet_embedding.pth')
    np.save('loss_history.npy', np.array(loss_history))

if __name__ == "__main__":
    main()
