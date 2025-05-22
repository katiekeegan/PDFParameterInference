import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import numpy as np
from models import PointNetPMA, InferenceNet
from utils import generate_data, H5Dataset, EventDataset, precompute_features_and_latents_to_disk, get_optimizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args, xs, thetas, pointnet_model):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    latent_dim = args.latent_dim
    latent_path = os.path.abspath(args.latent_path)
    if rank == 0:
        if os.path.exists(latent_path):
            os.remove(latent_path)
        precompute_features_and_latents_to_disk(pointnet_model, xs, thetas, latent_path, chunk_size=args.chunk_size)
    dist.barrier()

    dataset = H5Dataset(latent_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=EventDataset.collate_fn,
        num_workers=0,
        pin_memory=True
    )

    inference_net = InferenceNet(embedding_dim=latent_dim).to(device)
    inference_net = torch.nn.parallel.DistributedDataParallel(inference_net, device_ids=[rank])
    optimizer = get_optimizer(inference_net, lr=args.lr)
    scaler = amp.GradScaler()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for latent_embeddings, true_params in dataloader:
            latent_embeddings = latent_embeddings.to(device, non_blocking=True)
            true_params = true_params.to(device, non_blocking=True)
            with amp.autocast(dtype=torch.float16):
                recon_theta = inference_net(latent_embeddings)
                param_mins = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                param_maxs = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
                norm_pred = (recon_theta - param_mins) / (param_maxs - param_mins)
                norm_true = (true_params - param_mins) / (param_maxs - param_mins)
                loss = F.mse_loss(norm_pred, norm_true)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader):.4f}")
            if epoch % 100 == 0:
                torch.save(inference_net.module.state_dict(), f"inference_net_epoch_{epoch}.pth")

    if rank == 0:
        torch.save(inference_net.module.state_dict(), "inference_net_final.pth")
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--num_events', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--latent_path', type=str, default='latent_features.h5')
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--nodes', type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thetas, xs = generate_data(args.num_samples, args.num_events, device=device)

    input_dim = 6
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=args.latent_dim, predict_theta=True)
    state_dict = torch.load('final_model.pth', map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    pointnet_model.load_state_dict(state_dict)
    pointnet_model.eval()

    world_size = args.gpus * args.nodes
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(train, args=(world_size, args, xs, thetas, pointnet_model), nprocs=args.gpus)

if __name__ == '__main__':
    main()