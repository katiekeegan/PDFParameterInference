import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

from models import PointNetPMA


class NTXentThetaContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, sim_threshold=0.15, dissim_threshold=0.35, clip_logits=30.0):
        super().__init__()
        self.temperature = temperature
        self.sim_threshold = sim_threshold
        self.dissim_threshold = dissim_threshold
        self.clip_logits = clip_logits

    def forward(self, embeddings, thetas):
        embeddings = F.normalize(embeddings, dim=-1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim_matrix = torch.clamp(sim_matrix, min=-self.clip_logits, max=self.clip_logits)

        theta_dists = torch.cdist(thetas, thetas) / (theta_dists.max() + 1e-8)
        pos_mask = (theta_dists < self.sim_threshold).float()
        neg_mask = (theta_dists > self.dissim_threshold).float()

        logits_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(logits_mask, float("-inf"))

        losses = []
        for i in range(sim_matrix.size(0)):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            selected = torch.cat([pos_idx, neg_idx])
            logits = sim_matrix[i][selected]
            labels = torch.zeros(1, dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), labels))

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=sim_matrix.device, requires_grad=True)


class SimplifiedDIS:
    def __init__(self, device=None, smear=False, smear_std=0.05):
        self.device = device
        self.smear = smear
        self.smear_std = smear_std

    def init(self, params):
        self.au, self.bu, self.ad, self.bd = [p.to(self.device) for p in params]

    def up(self, x):
        return (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(params)
        eps = 1e-6
        rand = lambda: torch.clamp(torch.rand(nevents, device=self.device), min=eps, max=1 - eps)
        smear_noise = lambda s: s + torch.randn_like(s) * (self.smear_std * s) if self.smear else s

        xs_p, xs_n = rand(), rand()
        sigma_p = smear_noise(4 * self.up(xs_p) + self.down(xs_p))
        sigma_n = smear_noise(4 * self.down(xs_n) + self.up(xs_n))
        return torch.stack([sigma_p, sigma_n], dim=-1)


class DISDataset(IterableDataset):
    def __init__(self, simulator, num_samples, num_events, rank, world_size, theta_dim=4):
        self.simulator = simulator
        self.num_samples = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_bounds = torch.tensor([[0.0, 5]] * theta_dim)

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}")
        self.simulator.device = device
        for _ in range(self.num_samples):
            theta = torch.rand(self.theta_bounds.size(0), device=device)
            theta = theta * (self.theta_bounds[:, 1] - self.theta_bounds[:, 0]) + self.theta_bounds[:, 0]
            x = self.simulator.sample(theta, self.num_events)
            x_feat = feature_engineering(x)
            yield theta.cpu(), x_feat.cpu()


def feature_engineering(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    log_features = torch.log1p(x)
    symlog = torch.sign(x) * torch.log1p(x.abs())

    ratios, diffs = [], []
    for i in range(x.shape[-1]):
        for j in range(i + 1, x.shape[-1]):
            ratio = torch.log1p((x[..., i] / (x[..., j] + 1e-8)).abs())
            diff = torch.log1p(x[..., i]) - torch.log1p(x[..., j])
            ratios.append(ratio.unsqueeze(-1))
            diffs.append(diff.unsqueeze(-1))

    return torch.cat([log_features, symlog] + ratios + diffs, dim=-1)


def contrastive_loss_fn(latent, theta, margin=1.0, scale=2.0):
    latent = F.normalize(latent, p=2, dim=-1)
    emb_dists = 1.0 - torch.mm(latent, latent.t())
    theta_dists = torch.cdist(theta, theta) / (theta_dists.max() + 1e-8)
    sim_mask = (theta_dists < 0.15).float()
    dissim_mask = (theta_dists > 0.35).float()

    sim_loss = (sim_mask * emb_dists).mean()
    dissim_loss = (dissim_mask * F.relu(margin - emb_dists)).mean()
    return (sim_loss + scale * dissim_loss) / (1 + scale)


def train(model, dataloader, epochs, lr, rank, wandb_enabled):
    device = next(model.parameters()).device
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    alpha1, alpha2 = 0.01, 0.01

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for theta, x in dataloader:
            theta, x = theta.to(device), x.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                latent = model(x)
                loss = contrastive_loss_fn(latent, theta)
                loss += alpha1 * latent.norm(p=2, dim=1).mean()
                z = latent - latent.mean(dim=0)
                cov = (z.T @ z) / (z.size(0) - 1)
                decorrelation = ((cov * (1 - torch.eye(cov.size(0), device=cov.device))) ** 2).sum()
                loss += alpha2 * decorrelation
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        if rank == 0:
            wandb.log({"epoch": epoch + 1, "loss": total_loss})
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    if rank == 0:
        torch.save(model.state_dict(), "final_model.pth")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    simulator = SimplifiedDIS(device=device)
    dummy_theta = torch.rand(4, device=device)
    dummy_x = simulator.sample(dummy_theta, args.num_events)
    input_dim = feature_engineering(dummy_x).shape[-1]

    dataset = DISDataset(simulator, args.num_samples, args.num_events, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    model = PointNetPMA(input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    if rank == 0 and args.wandb:
        wandb.init(project="quantom_cl", config=vars(args))

    train(model, dataloader, args.num_epochs, args.lr, rank, args.wandb)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_events', type=int, default=100000)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
