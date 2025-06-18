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
from models import *
import os

def precompute_features_and_latents_to_disk(pointnet_model, xs_tensor, thetas, output_path, chunk_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pointnet_model = pointnet_model.to(device)
    pointnet_model.eval()

    latent_dim = 256
    num_samples = xs_tensor.shape[0]
    print(f"[precompute] Saving HDF5 to: {output_path}")
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        latent_dset = f.create_dataset('latents', shape=(num_samples, latent_dim), dtype=np.float32,
                                       chunks=(chunk_size, latent_dim))
        theta_dset = f.create_dataset('thetas', data=thetas.cpu().numpy(), dtype=np.float32)

        for i in tqdm(range(0, num_samples, chunk_size)):
            xs_chunk = xs_tensor[i:i+chunk_size].cpu()  # stay on CPU to avoid OOM
            xs_engineered = advanced_feature_engineering(xs_chunk)  # now CPU safe
            xs_engineered = xs_engineered.to(device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                latent = pointnet_model(xs_engineered)
                latent = latent.squeeze(1).cpu().numpy()
            
            latent_dset[i:i+len(latent)] = latent

            del xs_chunk, xs_engineered, latent
            torch.cuda.empty_cache()

# Suppress numerical warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import h5py
import numpy as np
from tqdm import tqdm

def precompute_latents_to_disk(pointnet_model, xs_tensor, thetas, output_path, chunk_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pointnet_model = pointnet_model.to(device)
    pointnet_model.eval()
    
    with h5py.File(output_path, 'w') as f:
        latent_shape = (len(xs_tensor), 256)
        latent_dset = f.create_dataset('latents', shape=latent_shape, dtype=np.float32,
                                       chunks=(chunk_size, latent_shape[1]))
        
        theta_dset = f.create_dataset('thetas', data=thetas.cpu().numpy(), dtype=np.float32)

        for i in tqdm(range(0, len(xs_tensor), chunk_size)):
            chunk = xs_tensor[i:i+chunk_size].to(device)
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                latent = pointnet_model(chunk)
                latent = latent.squeeze(1).cpu().numpy()
            latent_dset[i:i+len(latent)] = latent
            del chunk, latent
            torch.cuda.empty_cache()
    
    return output_path

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


class InferenceNet(nn.Module):
    def __init__(self, embedding_dim, ouput_dim = 4, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)  # Output raw (unconstrained) parameters
        )
        self._init_weights()
        
        # Parameter ranges: [au, bu, ad, bd]
        self.param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.param_maxs = torch.tensor([5, 5, 5, 5])
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # Get raw network output
        params = self.net(z)
        
        # # Apply constraints to each parameter
        # params = torch.zeros_like(raw_params)
        #         #         [0.1, 5],
        #         # [-1, -0.5],
        #         # [0.1, 5],
        #         # [-1, -0.5],
        
        # # au (0.1, 5) - use sigmoid then scale
        # params[:, 0] = 0.1 + (5.0 - 0.1) * torch.sigmoid(raw_params[:, 0])
        
        # # bu (-1, -0.1) - use sigmoid then scale to negative range
        # params[:, 1] =-1.0 + (-0.5 - (-1.0)) * torch.sigmoid(raw_params[:, 1])
        
        # # ad (0.1, 5) - same as au
        # params[:, 2] = 0.1 + (5.0 - 0.1) * torch.sigmoid(raw_params[:, 2])
        
        # # bd (-1, -0.5) - similar to bu but different range
        # params[:, 3] = -1.0 + (-0.5 - (-1.0)) * torch.sigmoid(raw_params[:, 3])
        
        return params

# Feature Engineering with improved stability
def advanced_feature_engineering(xs_tensor):
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

# SimplifiedDIS class with improved numerical stability
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
        self.au = torch.clamp(params[0], min=0.1, max=10)
        self.bu = torch.clamp(params[1], min=0.1, max=10)
        self.Nd = 2
        self.ad = torch.clamp(params[2], min=0.1, max=10)
        self.bd = torch.clamp(params[3], min=0.1, max=10)

    def up(self, x):
        x = torch.clamp(x, min=1e-8, max=1-1e-8)
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        x = torch.clamp(x, min=1e-8, max=1-1e-8)
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(torch.tensor(params, device=self.device))

        xs_p = torch.rand(nevents, device=self.device)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0, posinf=1e8, neginf=0.0)

        xs_n = torch.rand(nevents, device=self.device)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0, posinf=1e8, neginf=0.0)

        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()

    def batch_up(self, params, xs):
        """Compute up for a batch of parameters"""
        self.au = params[:, 0]
        self.bu = params[:, 1]
        self.ad = params[:, 2]
        self.bd = params[:, 3]
        x_expanded = xs.unsqueeze(0)  # (1, num_x)
        x_clamped = torch.clamp(x_expanded, min=1e-8, max=1-1e-8)
        return self.Nu * (x_clamped ** self.au.unsqueeze(1)) * ((1 - x_clamped) ** self.bu.unsqueeze(1))
    
    def batch_down(self, params, xs):
        """Compute down for a batch of parameters"""
        self.au = params[:, 0]
        self.bu = params[:, 1]
        self.ad = params[:, 2]
        self.bd = params[:, 3]
        x_expanded = xs.unsqueeze(0)  # (1, num_x)
        x_clamped = torch.clamp(x_expanded, min=1e-8, max=1-1e-8)
        return self.Nd * (x_clamped ** self.ad.unsqueeze(1)) * ((1 - x_clamped) ** self.bd.unsqueeze(1))

# Data Generation with improved stability
def generate_data(num_samples, num_events, problem, theta_dim=4, x_dim=2, device=torch.device("cpu")):
    if problem == 'simplified_dis':
        simulator = SimplifiedDIS(device)
        theta_dim = 4 # [au, bu, ad, bd]
        ranges = [(0.0, 5), (0.0, 5), (0.0, 5), (0.0, 5)]  # Example ranges
    elif problem == 'realistic_dis':
        simulator = RealisticDISSimulator(device)
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

def get_optimizer(model, lr):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

def get_scheduler(optimizer, epochs):
    return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                       total_steps=epochs, 
                                       pct_start=0.3)

def setup(rank, world_size):
    import socket
    from contextlib import closing
    
    # Find a free port
    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    
    port = find_free_port()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def train(rank, world_size, args, xs, thetas, pointnet_model):
    # Setup distributed training
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Initialize models with DDP
    latent_dim = 256
    def latent_fn(event):
        # Automatically use the same device as the model
        model_device = next(pointnet_model.parameters()).device
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            event = event.to(model_device)
            latent, _ = pointnet_model(event)
            latent = latent.squeeze(0).cpu()
            return latent
    
    # Create dataset with distributed sampler
    # dataset = EventDataset(xs_tensor_engineered, thetas, latent_fn)
    # Compute latents in chunks (saves memory)
    latent_path = os.path.abspath('latent_features.h5')

    # Only rank 0 creates the file
    if rank == 0:
        if os.path.exists(latent_path):
            os.remove(latent_path)
        precompute_features_and_latents_to_disk(pointnet_model, xs, thetas, latent_path, chunk_size=2)

    # All ranks wait until the file is ready
    dist.barrier()
    inference_net = InferenceNet(embedding_dim=latent_dim).to(device)
    inference_net = DDP(inference_net, device_ids=[rank])
    dataset = H5Dataset(latent_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        collate_fn=EventDataset.collate_fn,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
                          # Initialize dataset with streaming
# dataset = EventDataset(xs_tensor_engineered, thetas, pointnet_model, device)

    # # Use custom collate function in DataLoader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=128,  # Can be larger now since we're not storing all latents
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,
    #     collate_fn=EventDataset.collate_fn  # Important!
    # )
    
    # Optimizer and scheduler
    optimizer = get_optimizer(inference_net, lr=1e-4)
    # scheduler = get_scheduler(optimizer, epochs=args.epochs)
    
    # Mixed precision and gradient scaling
    scaler = amp.GradScaler()
    if rank == 0:
        recon_losses = []
        normalized_losses = []
        total_losses = []

    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        epoch_loss = 0.0
        
        for batch_idx, (latent_embeddings, true_params) in enumerate(dataloader):
            torch.cuda.empty_cache()
            latent_embeddings = latent_embeddings.to(device, non_blocking=True)
            true_params = true_params.to(device, non_blocking=True)
            with amp.autocast(dtype=torch.float16):
                # Forward pass
                recon_theta = inference_net(latent_embeddings)
            
                recon_theta = inference_net(latent_embeddings)
                param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
                normalized_pred = (recon_theta - param_mins) / (param_maxs - param_mins)
                normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
                loss = F.mse_loss(normalized_pred, normalized_true)
                # kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                # loss = recon_loss + 0.5 * kl_loss # + 0.1* physics_loss  # beta-VAE style
            
            # Backward pass with gradient scaling
            optimizer.zero_grad(set_to_none=True)
            # print(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()
            
            epoch_loss += loss.item()
        
        # Only print from rank 0
        if rank == 0:
            print(f'Epoch: {epoch}, Loss: {epoch_loss/len(dataloader)}')
            total_losses.append(loss.item())
            if epoch % 100 == 0:
                torch.save(inference_net.module.state_dict(), f"trained_inference_net_epoch_{epoch}.pth")
    
    # Final save from rank 0
    if rank == 0:
        torch.save(inference_net.module.state_dict(), "final_inference_net.pth")
        np.save("loss_total.npy", np.array(total_losses))
    
    cleanup()

def setup(rank, world_size):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def precompute_latents_chunked(pointnet_model, xs_tensor, chunk_size=32, device='cuda'):
    """
    Compute latent features in memory-friendly chunks and normalize them.
    
    Returns:
        latents_normalized: Tensor of normalized latent features
        z_mean: Tensor of latent feature means (shape: [latent_dim])
        z_std: Tensor of latent feature stds (shape: [latent_dim])
    """
    pointnet_model = pointnet_model.to(device)
    pointnet_model.eval()
    
    latents = []
    with torch.no_grad():
        for i in range(0, len(xs_tensor), chunk_size):
            chunk = xs_tensor[i:i+chunk_size].to(device)
            latent, _ = pointnet_model(chunk).squeeze(1).cpu()  # shape: [B, latent_dim]
            latents.append(latent)
            del chunk, latent
            torch.cuda.empty_cache()
    
    latents = torch.cat(latents, dim=0)  # shape: [N, latent_dim]
    
    # Compute normalization stats
    z_mean = latents.mean(dim=0, keepdim=True)  # shape: [1, latent_dim]
    z_std = latents.std(dim=0, keepdim=True) + 1e-6  # Avoid divide-by-zero
    
    # Normalize
    latents_normalized = (latents - z_mean) / z_std

    return latents_normalized, z_mean.squeeze(), z_std.squeeze()
def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--nodes', type=int, default=1)  # For multi-node support
    parser.add_argument('--problem', type=str, default='simplified_dis', choices=['simplified_dis', 'realistic_dis'])
    args = parser.parse_args()
    
    # Common setup for both single and multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 256
    
    # Sample and prepare data (do this once, not in both branches)
    num_samples = 2000
    num_events = 1000000
    thetas, xs = generate_data(num_samples, num_events, problem=args.problem, device=device)
    input_dim = 6
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=latent_dim, predict_theta=True)
    # pointnet_model.load_state_dict(torch.load('pointnet_embedding_latent_dim_1024.pth', map_location='cpu'))
    state_dict = torch.load('final_model.pth', map_location='cpu')
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict(state_dict)
    pointnet_model.eval()

    def latent_fn(event):
        # Automatically use the same device as the model
        model_device = next(pointnet_model.parameters()).device
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            event = event.to(model_device)
            latent, _ = pointnet_model(event)
            latent = latent.squeeze(0).cpu()
            return latent
    
    if args.gpus > 1:
        # Multi-GPU setup
        world_size = args.gpus * args.nodes
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        torch.multiprocessing.spawn(
            train,
            args=(world_size, args, xs, thetas, pointnet_model),
            nprocs=args.gpus
        )
    else:
        # Single GPU setup
        pointnet_model = pointnet_model.to(device)
        # Initialize dataset with streaming
        # dataset = EventDataset(xs_tensor_engineered, thetas, latent_fn)
        # Compute latents in chunks (saves memory)
        latent_path = 'latent_features.h5'
        if not os.path.exists(latent_path):
            xs_tensor_engineered = advanced_feature_engineering(xs)
    
            # Initialize PointNetEmbedding model (do this once)
            input_dim = xs_tensor_engineered.shape[-1]
            precompute_latents_to_disk(pointnet_model, 
                                    xs_tensor_engineered,
                                    latent_path,
                                    chunk_size=8)
            del xs_tensor_engineered
            del xs

        dataset = H5Dataset(latent_path, thetas)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, 
                                collate_fn=EventDataset.collate_fn, num_workers=0, pin_memory=True, persistent_workers=False)
        
        inference_net = InferenceNet(embedding_dim=latent_dim, output_dim=thetas.size(-1)).to(device)
        optimizer = get_optimizer(inference_net, lr=1e-4)
        # scheduler = get_scheduler(optimizer, epochs=args.epochs)
        scaler = amp.GradScaler()
        
        # Training loop
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for batch_idx, (latent_embeddings, true_params) in enumerate(dataloader):
                torch.cuda.empty_cache()
                latent_embeddings = latent_embeddings.to(device, non_blocking=True)
                true_params = true_params.to(device, non_blocking=True)
                
                with amp.autocast(dtype=torch.float16):
                    pred_params = inference_net(latent_embeddings)
                    xs = torch.rand(1000, device=device)
                    
                    # # Use the batch processing methods we added earlier
                    # simulator = SimplifiedDIS(device=device)
                    # true_u = simulator.batch_up(true_params, xs)
                    # true_d = simulator.batch_down(true_params, xs)
                    # pred_u = simulator.batch_up(pred_params, xs)
                    # pred_d = simulator.batch_down(pred_params, xs)
                    
                    # # Clamp and compute loss
                    # pred_u = torch.clamp(pred_u, min=1e-8, max=1e8)
                    # pred_d = torch.clamp(pred_d, min=1e-8, max=1e8)
                    # true_u = torch.clamp(true_u, min=1e-8, max=1e8)
                    # true_d = torch.clamp(true_d, min=1e-8, max=1e8)
                    
                    # loss_u = ((pred_u - true_u) / (true_u + 1e-8)).pow(2).mean()
                    # loss_d = ((pred_d - true_d) / (true_d + 1e-8)).pow(2).mean()
                    # loss = (loss_u + loss_d) / 2
                    recon_theta = inference_net(latent_embeddings)
                    recon_loss = F.mse_loss(recon_theta, true_params)
                    param_mins = torch.tensor([0.1, -1.0, 0.1, -1.0], device=device)
                    param_maxs = torch.tensor([5.0, -0.5, 5.0, -0.5], device=device)
                    normalized_pred = (recon_theta - param_mins) / (param_maxs - param_mins)
                    normalized_true = (true_params - param_mins) / (param_maxs - param_mins)
                    loss = F.mse_loss(normalized_pred, normalized_true)
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                
                epoch_loss += loss.item()
            
            print(f'Epoch: {epoch}, Loss: {epoch_loss/len(dataloader)}')
            if epoch % 10 == 0:
                torch.save(inference_net.state_dict(), "trained_inference_net.pth")
        
        torch.save(inference_net.state_dict(), "final_inference_net.pth")

if __name__ == "__main__":
    main()