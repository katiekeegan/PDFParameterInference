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
from datasets import *
from utils import *

from models import PointNetPMA

from simulator import RealisticDIS, Gaussian2DSimulator

class SimplifiedDIS:
    def __init__(self, device=None, smear=False, smear_std=0.05):
        self.device = device
        self.smear = smear
        self.smear_std = smear_std
        self.Nu = 1
        self.Nd = 2
        self.au, self.bu, self.ad, self.bd = None, None, None, None

    def init(self, params):
        self.au, self.bu, self.ad, self.bd = [p.to(self.device) for p in params]

    def up(self, x):
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(params)
        eps = 1e-6
        rand = lambda: torch.clamp(torch.rand(nevents, device=self.device), min=eps, max=1 - eps)
        smear_noise = lambda s: s + torch.randn_like(s) * (self.smear_std * s) if self.smear else s

        xs_p, xs_n = rand(), rand()
        sigma_p = smear_noise(4 * self.up(xs_p) + self.down(xs_p))
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0, posinf=1e8, neginf=0.0)
        sigma_n = smear_noise(4 * self.down(xs_n) + self.up(xs_n))
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0, posinf=1e8, neginf=0.0)
        return torch.stack([sigma_p, sigma_n], dim=-1)

def contrastive_loss_fn(z, theta, temperature=0.1, sim_threshold=0.15, dissim_threshold=0.35, margin=1.0, scale=2.0):
    # Normalize latent set-level embeddings
    z = F.normalize(z, dim=-1)  # [B, d]
    
    # Cosine similarity matrix
    sim_matrix = torch.matmul(z, z.T)  # [B, B]
    
    # Compute parameter distances between theta vectors
    theta_dists = torch.cdist(theta, theta, p=2)
    theta_dists = theta_dists / (theta_dists.max() + 1e-8)

    # Define positive/negative masks
    pos_mask = (theta_dists < sim_threshold).float()
    neg_mask = (theta_dists > dissim_threshold).float()

    # Remove self-pairs
    diag_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
    pos_mask *= (1 - diag_mask)
    neg_mask *= (1 - diag_mask)

    # Compute distances for positive pairs (contrastive loss part 1)
    pos_loss = (1 - sim_matrix) * pos_mask
    pos_loss = pos_loss.sum() / (pos_mask.sum() + 1e-8)

    # For negative pairs, use hinge loss with margin
    neg_margin = F.relu(margin - (1 - sim_matrix)) * neg_mask
    neg_loss = neg_margin.sum() / (neg_mask.sum() + 1e-8)

    total = (pos_loss + scale * neg_loss) / (1 + scale)
    return total

def triplet_theta_contrastive_loss(z, theta, margin=0.5, sim_threshold=0.05, dissim_threshold=0.1):
    """
    z: [B, d] - normalized embeddings
    theta: [B, d_theta] - ground-truth parameters
    """
    z = F.normalize(z, dim=-1)
    theta_dists = torch.cdist(theta, theta, p=2)
    theta_dists = theta_dists / (theta_dists.max() + 1e-8)

    B = z.size(0)
    losses = []

    for i in range(B):
        anchor = z[i]
        pos_mask = (theta_dists[i] < sim_threshold)
        neg_mask = (theta_dists[i] > dissim_threshold)

        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        # Choose hardest positive (farthest in embedding space)
        pos_embs = z[pos_indices]
        pos_dists = F.pairwise_distance(anchor.unsqueeze(0), pos_embs)
        hardest_pos = pos_embs[pos_dists.argmax()]

        # Choose hardest negative (closest in embedding space)
        neg_embs = z[neg_indices]
        neg_dists = F.pairwise_distance(anchor.unsqueeze(0), neg_embs)
        hardest_neg = neg_embs[neg_dists.argmin()]

        # Triplet loss: max(d(anchor, pos) - d(anchor, neg) + margin, 0)
        triplet_loss = F.triplet_margin_loss(
            anchor.unsqueeze(0),
            hardest_pos.unsqueeze(0),
            hardest_neg.unsqueeze(0),
            margin=margin,
            reduction='none'
        )
        losses.append(triplet_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    return torch.cat(losses).mean()

def train(model, dataloader, epochs, lr, rank, wandb_enabled, output_dir):
    device = next(model.parameters()).device
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    alpha1, alpha2 = 0.01, 0.01

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for theta, x_sets in dataloader:
            B, n_repeat, num_points, feat_dim = x_sets.shape
            x_sets = x_sets.view(B * n_repeat, num_points, feat_dim)  # Flatten
            theta = theta.repeat_interleave(n_repeat, dim=0)

            theta, x_sets = theta.to(device), x_sets.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                latent = model(x_sets)
                contrastive = triplet_theta_contrastive_loss(latent, theta)

                l2_reg = latent.norm(p=2, dim=1).mean()
                z = latent - latent.mean(dim=0)
                cov = (z.T @ z) / (z.size(0) - 1)
                decorrelation = ((cov * (1 - torch.eye(cov.size(0), device=cov.device))) ** 2).sum()

                loss = contrastive + alpha1 * l2_reg + alpha2 * decorrelation

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        if rank == 0:
            if wandb_enabled:
                wandb.log({"epoch": epoch + 1, "loss": total_loss})
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(output_dir, "most_recent_model.pth"))

    if rank == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))




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

    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.problem}_latent{args.latent_dim}_ns_{args.num_samples}_ne_{args.num_events}"

    # Define output directory and create it
    output_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.problem == 'simplified_dis':
        simulator = SimplifiedDIS(device=device)
        dataset = DISDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        input_dim = 4
    elif args.problem == 'realistic_dis':
        simulator = RealisticDIS(device=device, smear=True, smear_std=0.05)
        print("Simulator constructed!")
        dataset = RealisticDISDataset(simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=log_feature_engineering)
        print("Dataset created!")
        input_dim = 6
    elif args.problem == 'gaussian':
        simulator = Gaussian2DSimulator(device=device)
        dataset = Gaussian2DDataset(
        simulator, args.num_samples, args.num_events, rank, world_size
        )
        input_dim = 2

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    if not (args.problem == 'gaussian'):
        dummy_theta = torch.rand(input_dim, device=device)
        dummy_x = simulator.sample(dummy_theta, args.num_events)
        input_dim = log_feature_engineering(dummy_x).shape[-1]

    model = PointNetPMA(input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    if rank == 0 and args.wandb:
        wandb.init(project="quantom_cl", name=args.experiment_name, config=vars(args))

    train(model, dataloader, args.num_epochs, args.lr, rank, args.wandb, output_dir)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_events', type=int, default=500000)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--problem', type=str, default='simplified_dis')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default=None, help='Unique name for this ablation run')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
