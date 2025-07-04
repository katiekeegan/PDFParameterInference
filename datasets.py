import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import *
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.cuda.amp as amp
import warnings
import os
import h5py
import numpy as np
from tqdm import tqdm
from simulator import SimplifiedDIS, RealisticDIS, up, down, advanced_feature_engineering
from utils import log_feature_engineering

def generate_gaussian2d_dataset(n_samples, n_events, device=None):
    """
    Generate a dataset of n_samples parameter vectors and n_events events each.
    Parameter vector: [mu_x, mu_y, sigma_x, sigma_y, rho]
    """
    device = device or torch.device("cpu")
    # Example: fixed ranges for parameters, can be changed as needed
    mus = torch.empty((n_samples, 2)).uniform_(-2, 2)
    sigmas = torch.empty((n_samples, 2)).uniform_(0.5, 2.0)
    rhos = torch.empty((n_samples, 1)).uniform_(-0.8, 0.8)
    thetas = torch.cat([mus, sigmas, rhos], dim=1).to(device)

    simulator = Gaussian2DSimulator(device=device)
    xs = []
    for i in range(n_samples):
        x = simulator.sample(thetas[i], nevents=n_events)
        xs.append(x)
    xs = torch.stack(xs, dim=0)  # (n_samples, n_events, 2)
    return thetas, xs

class Gaussian2DDataset(IterableDataset):
    def __init__(
        self,
        simulator,
        num_samples,
        num_events,
        rank,
        world_size,
        theta_dim=5,        # mu_x, mu_y, sigma_x, sigma_y, rho
        theta_bounds=None,
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
            # [mu_x, mu_y, sigma_x, sigma_y, rho]
            self.theta_bounds = torch.tensor([
                [-2.0, 2.0],   # mu_x
                [-2.0, 2.0],   # mu_y
                [0.5, 2.0],    # sigma_x
                [0.5, 2.0],    # sigma_y
                [-0.8, 0.8],   # rho
            ])
        else:
            self.theta_bounds = torch.tensor(theta_bounds)

        self.feature_engineering = feature_engineering

    def __iter__(self):
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.simulator.device = device
        theta_bounds = self.theta_bounds.to(device)

        for _ in range(self.num_samples):
            theta = torch.rand(self.theta_dim, device=device)
            theta = theta * (theta_bounds[:, 1] - theta_bounds[:, 0]) + theta_bounds[:, 0]

            xs = []
            for _ in range(self.n_repeat):
                x = self.simulator.sample(theta, self.num_events)  # shape: [num_events, 2]
                if self.feature_engineering is not None:
                    x = self.feature_engineering(x)
                xs.append(x.cpu())
            yield theta.cpu(), torch.stack(xs)  # shape: [n_repeat, num_events, 2]

class H5Dataset(Dataset):
    def __init__(self, latent_path):
        self.latent_file = h5py.File(latent_path, 'r')
        self.latents = self.latent_file['latents']
        self.thetas = self.latent_file['thetas']
        self.indices = np.arange(len(self.latents), dtype=np.int64)  # correct indexing

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # PyTorch sometimes gives batched index arrays → ensure scalar access
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = idx[0]  # Only support 1-sample fetches (because we're using collate_fn)
        real_idx = self.indices[int(idx)]
        latent = torch.from_numpy(self.latents[real_idx])
        theta = torch.from_numpy(self.thetas[real_idx])
        return latent, theta

    def __del__(self):
        try:
            self.latent_file.close()
        except Exception:
            pass

class EventDataset(Dataset):
    def __init__(self, event_data, param_data, latent_fn):
        self.event_data = event_data
        self.param_data = param_data
        self.latent_fn = latent_fn

    def __len__(self):
        return len(self.param_data)

    def __getitem__(self, idx):
        latent = self.latent_fn(self.event_data[idx].unsqueeze(0))  # compute on-the-fly in main process
        return latent, self.param_data[idx]

    @staticmethod
    def collate_fn(batch):
        latents, params = zip(*batch)
        return torch.stack(latents), torch.stack(params)

# Data Generation with improved stability
def generate_data(num_samples, num_events, problem, theta_dim=4, x_dim=2, device=torch.device("cpu")):
    if problem == 'simplified_dis':
        simulator = SimplifiedDIS(device)
        theta_dim = 4 # [au, bu, ad, bd]
        ranges = [(0.0, 5), (0.0, 5), (0.0, 5), (0.0, 5)]  # Example ranges
    elif problem == 'realistic_dis':
        simulator = RealisticDIS(device)
        theta_dim = 6
        ranges = [
                (-2.0, 2.0),   # logA0
                (-1.0, 1.0),   # delta
                (0.0, 5.0),    # a
                (0.0, 10.0),   # b
                (-5.0, 5.0),   # c
                (-5.0, 5.0),   # d
        ]
    thetas = np.column_stack([np.random.uniform(low, high, size=num_samples) for low, high in ranges])
    xs = np.array([simulator.sample(theta, num_events).cpu().numpy() for theta in thetas]) + 1e-8
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32, device=device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    return thetas_tensor, xs_tensor

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
                xs.append(log_feature_engineering(x).cpu())

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