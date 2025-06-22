import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
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