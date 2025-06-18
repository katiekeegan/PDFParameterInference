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

class RealisticDISSimulator:
    def __init__(self, device=None, smear=True, smear_std=0.05):
        self.device = device or torch.device("cpu")
        self.smear = smear
        self.smear_std = smear_std
        self.Q0_squared = 1.0  # GeV^2 reference scale
        self.params = None

    def __call__(self, params, nevents=1000):
        return self.sample(params, nevents)

    def init(self, params):
        # Accepts raw list or tensor of 6 params: [logA0, delta, a, b, c, d]
        p = torch.tensor(params, dtype=torch.float32, device=self.device)
        self.logA0 = p[0]
        self.delta = p[1]
        self.a = p[2]
        self.b = p[3]
        self.c = p[4]
        self.d = p[5]

    def q(self, x, Q2):
        A0 = torch.exp(self.logA0)
        scale_factor = (Q2 / self.Q0_squared).clamp(min=1e-6)
        A_Q2 = A0 * scale_factor ** self.delta
        shape = x.clamp(min=1e-6, max=1.0)**self.a * (1 - x.clamp(min=0.0, max=1.0))**self.b
        poly = 1 + self.c * x + self.d * x**2
        shape = shape * poly.clamp(min=1e-6)  # avoid negative polynomial tail
        return A_Q2 * shape

    def F2(self, x, Q2):
        return x * self.q(x, Q2)

    def sample(self, params, nevents=1000, x_range=(1e-3, 0.9), Q2_range=(1.0, 1000.0)):
        self.init(params)

        # Sample x ~ Uniform, Q2 ~ LogUniform
        x = torch.rand(nevents, device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
        logQ2 = torch.rand(nevents, device=self.device) * (
            np.log10(Q2_range[1]) - np.log10(Q2_range[0])
        ) + np.log10(Q2_range[0])
        Q2 = 10 ** logQ2

        f2 = self.F2(x, Q2)

        if self.smear:
            noise = torch.randn_like(f2) * (self.smear_std * f2)
            f2 = f2 + noise
            f2f = f2.clamp(min=1e-6)

        return torch.stack([x, Q2, f2], dim=1)  # shape: [nevents, 3]s

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
    def __init__(self, simulator, num_samples, num_events, rank, world_size, theta_dim=4, n_repeat=2):
        self.simulator = simulator
        self.num_samples = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_bounds = torch.tensor([[0.0, 5]] * theta_dim)
        self.n_repeat = n_repeat

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}")
        self.simulator.device = device
        theta_bounds = self.theta_bounds.to(device)

        for _ in range(self.num_samples):
            theta = torch.rand(theta_bounds.size(0), device=device)
            theta = theta * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]
            xs = []
            for _ in range(self.n_repeat):
                x = self.simulator.sample(theta, self.num_events)
                xs.append(feature_engineering(x).cpu())

            yield theta.cpu(), torch.stack(xs)  # [n_repeat, num_points, feature_dim]

class RealisticDISDataset(IterableDataset):
    def __init__(
        self,
        simulator,
        num_samples,
        num_events,
        rank,
        world_size,
        theta_dim=6,        # logA0, delta, a, b, c, d
        theta_bounds=None,  # custom bounds (optional)
        n_repeat=2,
        feature_engineering=None,
    ):
        self.simulator = simulator
        self.num_samples = num_samples // world_size
        self.num_events = num_events
        self.rank = rank
        self.world_size = world_size
        self.theta_dim = theta_dim
        self.n_repeat = n_repeat

        if theta_bounds is None:
            # Default bounds per parameter: adjust based on physical priors
            self.theta_bounds = torch.tensor([
                [-2.0, 2.0],   # logA0
                [-1.0, 1.0],   # delta
                [0.0, 5.0],    # a
                [0.0, 10.0],   # b
                [-5.0, 5.0],   # c
                [-5.0, 5.0],   # d
            ])
        else:
            self.theta_bounds = torch.tensor(theta_bounds)

        self.feature_engineering = feature_engineering# or (lambda x: x)

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.simulator.device = device
        theta_bounds = self.theta_bounds.to(device)

        for _ in range(self.num_samples):
            theta = torch.rand(self.theta_dim, device=device)
            theta = theta * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]

            xs = []
            for _ in range(self.n_repeat):
                # Sample events from the simulator
                x = self.simulator.sample(theta, self.num_events)  # shape: [num_events, 3]
                xs.append(self.feature_engineering(x).cpu())       # e.g., log(x), normalize, etc.

            yield theta.cpu(), torch.stack(xs)  # shape: [n_repeat, num_events, feature_dim]

# Feature Engineering with improved stability
def feature_engineering(xs_tensor):
    # Basic features with clamping for numerical stability
    xs_clamped = torch.clamp(xs_tensor, min=1e-8, max=1e8)
    del xs_tensor
    log_features = torch.log1p(xs_clamped)
    symlog_features = torch.sign(xs_clamped) * torch.log1p(xs_clamped.abs())

    # Pairwise features with vectorized operations
    n_features = xs_clamped.shape[-1]
    # del xs_tensor
    combinations = torch.combinations(torch.arange(n_features), r=2)
    i, j = combinations[:, 0], combinations[:, 1]
    del combinations
    # Safe division with clamping
    ratio = xs_clamped[..., i] / (xs_clamped[..., j] + 1e-8)
    ratio_features = torch.log1p(ratio.abs())
    del ratio
    
    diff_features = torch.log1p(xs_clamped[..., i]) - torch.log1p(xs_clamped[..., j])
    del xs_clamped
    data = torch.cat([log_features, symlog_features, ratio_features, diff_features], dim=-1)
    return data

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

def train(model, dataloader, epochs, lr, rank, wandb_enabled):
    device = next(model.parameters()).device
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    alpha1, alpha2 = 0.01, 0.01

    if wandb_enabled and rank == 0:
        import wandb
        wandb.init(project="quantom_cl")

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
                latent = model(x_sets)  # [B*n_repeat, latent_dim]
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
    if args.problem == 'simplified_dis':
        simulator = SimplifiedDIS(device=device)
        dataset = DISDataset(simulator, args.num_samples, args.num_events, rank, world_size)
        input_dim = 4
    elif args.problem == 'realistic_dis':
        simulator = RealisticDISSimulator(device=device, smear=True, smear_std=0.05)
        print("Simulator constructed!")
        dataset = RealisticDISDataset(simulator, args.num_samples, args.num_events, rank, world_size, feature_engineering=feature_engineering)
        print("Dataset created!")
        input_dim = 6
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    dummy_theta = torch.rand(input_dim, device=device)
    dummy_x = simulator.sample(dummy_theta, args.num_events)
    print(f"dummy_theta: {dummy_theta}")
    print(f"dummy_x: {dummy_x}")
    input_dim = feature_engineering(dummy_x).shape[-1]
    model = PointNetPMA(input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
 
    if rank == 0 and args.wandb:
        wandb.init(project="quantom_cl", config=vars(args))

    train(model, dataloader, args.num_epochs, args.lr, rank, True)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_events', type=int, default=100000)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--problem', type=str, default='simplified_dis')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
