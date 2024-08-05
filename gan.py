import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import *
from sampler import *
from plotting import *

def train_loop(data, Generator, Discriminator, D_optimizer, args, criterion, losses, device):
    optimizer.zero_grad()
    z = torch.randn(args.batch_size, 1, args.noise_dim)
    gen_params = Generator(z)
    # gen_means, gen_stds = gen_params.chunk(2, dim=-1)
    # mask = gen_stds < 0  # Create a mask for negative entries
    # gen_stds[mask] = 10  # Replace negative entries with 'a'
    # gen_params = torch.cat((gen_means, gen_stds), dim=-1)
    print(gen_params)
    gen_data = torch.Tensor([])

    for i in range(args.batch_size):
        gen_data_i = (
            generate_synthetic_data(
                gen_params[i, :], args.sample_size, device, option=args.distribution
            )
            .float()
        )
        gen_data_i.requires_grad = True
        print(gen_data_i.size())
        gen_data = torch.cat((gen_data, gen_data_i.unsqueeze(0)), dim=0)
    loss = sinkhorn_distance(gen_data, data)# + 2* torch.mean(torch.clamp(-gen_data[:,:,-1], min=0)**2)
    # Compute batched optimal transport loss
    # loss = batched_optimal_transport_loss(gen_data, data)
    # torch.nn.utils.clip_grad_norm_(Generator.parameters(), 1.0)
    loss.backward()
    losses.append(loss.detach().numpy())
    D_optimizer.step()

    print(loss)
    fig1 = plt.figure(1)
    plt.clf()
    # ax = fig1.add_subplot(1, 1, 1)
    plot_dist(gen_data[0, ...].detach().numpy(), fig = fig1, plot_Gaussian=False,plot_other_data=True, other_data =data[0, ...].detach().numpy())
    # ax.hist(gen_data[0, ...].unsqueeze(-1).detach().numpy(), alpha=0.5)
    # ax.hist(data[0, ...].unsqueeze(-1).detach().numpy(), alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)
    fig2 = plt.figure(2)
    plt.clf()
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(losses, alpha=0.5, label="Step Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)


def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description="Diffusion Denoising Model for Probability Distribution Parameter Estimation"
    )
    parser.add_argument(
        "--noise-dim", type=int, default=32, help="Noise dimension size (default: 100)"
    )
    parser.add_argument(
        "--distribution",
        default="mixed",
        help="True data distribution (default: exp)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1024,
        help="input sample size for training (default: 1024)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="input batch size for training (default: 3)",
    )
    parser.add_argument(
        "--n-true-events",
        type=int,
        default=102400,
        help="Number of true events for training (default: 10000)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="generator",
        help="Filename for results",
    )
    args = parser.parse_args()

    # Hyperparameters
    alpha = 0.99999

    # Set true parameters
    if args.distribution == "normal":
        mu = 1.6  # true mean of Gaussian
        std = 0.8  # true std of Gaussian
        tparams = torch.tensor([mu, std])
    elif args.distribution == "exp":
        rate = 2  # true rate of exponential distribution
        tparams = torch.tensor([rate])
    elif args.distribution == "mixture":
        rate = 2  # true rate of exponential distribution
        tparams = torch.tensor([rate])
    elif args.distribution == "multidimensional":
        rate = 2  # true rate of exponential distribution
        tparams = torch.tensor([rate])
    elif args.distribution == "2D":
        rate = 2  # true rate of exponential distribution
        tparams = torch.tensor([rate])
    elif args.distribution == "mixed":
        tparams = torch.abs(torch.cat([torch.randn(5, 2).ravel(), torch.rand(5, 2).ravel()]))


    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    print("Device: {}".format(device))
    param_dims = sum(tensor.numel() for tensor in tparams)
    Generator = ConvolutionalGAN(args.noise_dim, param_dims).to(device)
    Discriminator = DiscriminatorNetwork(args.sample_size).to(device)
    # Define loss function and optimizers

    criterion = nn.MSELoss()
    G_optimizer = optim.Adam(Discriminator.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(Generator.parameters(), lr=args.lr)

    # Generate true events dataset
    dataset = generate_synthetic_data(
        tparams.unsqueeze(0), args.n_true_events, device, option=args.distribution
    ).squeeze()

    # Initialize losses
    losses = []

    shuffle_data = True
    samples_data_loader = DataLoader(
        dataset=dataset, batch_size=args.sample_size, shuffle=shuffle_data
    )
    dataset_with_subsamples = torch.tensor([])
    for tevents in samples_data_loader:
        dataset_with_subsamples = torch.cat(
            (dataset_with_subsamples, tevents.unsqueeze(0)), dim=0
        )

    data_loader = DataLoader(
        dataset=dataset_with_subsamples,
        batch_size=args.batch_size,
        shuffle=shuffle_data,
    )
    Generator.train()
    for epoch in range(args.epochs):
        for data in data_loader:
            print(epoch)
            train_loop(data, Generator, Discriminator, D_optimizer, args, criterion, losses, device)

    torch.save(Generator.state_dict(), str(args.filename) + "model.pt")


if __name__ == "__main__":
    main()

def closure():
    for _ in range(1):
        # put both G and D in evaluation mode
        G.eval()
        D.eval()

        # check how D does with true data
        tlabels = torch.ones(tevents.size(0), 1) # label 1 for true samples
        D.eval()
        toutput = D(tevents)

        # normalizing toutput so that the discriminator input is in [0,1]
        toutput_min = toutput.min()
        toutput_max = toutput.max()
        toutput_normalized = (toutput - toutput_min) / (toutput_max - toutput_min)

        # calculate loss on true events
        tloss = D.loss_fn(toutput_normalized ,tlabels ,args)

        # generate fake parameters and fake events
        noise = _generate_noise(args, device, length = args.batch_size)
        fparams_dist = G(noise)
        fparams_mean = torch.mean(fparams_dist, axis=0)
        fevents = T_fn(fparams_mean ,args.batch_size, device, args.distribution)

        # check how D does with fake data generated from fake parameters
        flabels = torch.zeros(fevents.size(0), 1)  # label 0 for fake samples
        foutput = D(fevents)

        # normalizing foutput so that the discriminator input is in [0,1]
        foutput_min = foutput.min()
        foutput_max = foutput.max()
        foutput_normalized = (foutput - foutput_min) / (foutput_max - foutput_min)

        # calculate loss on fake events
        floss = D.loss_fn(foutput_normalized, flabels ,args)

        # add true and fake losses (maybe this should be a weighted sum?)
        D_loss = tloss + floss
        log_batch['D-loss'][batch_idx] = D_loss.item()
        D.train()
        D_optimizer.zero_grad()
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=0.5)
        D_optimizer.step()
        # for param in D.parameters():
        #     param.data.clamp_(-0.01, 0.01)

    # taking another training step with generator

    # put both G and D in evaluation mode
    G.eval()
    D.eval()

    # generate fake parameters and fake events
    noise = _generate_noise(args, device, length=args.batch_size)
    fparams_dist = G(noise)
    fparams_mean = torch.mean(fparams_dist, axis=0)
    fevents = T_fn(fparams_mean, args.batch_size, device, args.distribution)

    # obtain fake loss
    foutput = D(fevents)
    flabels = torch.zeros(fevents.size(0), 1)
    G_loss = D.loss_fn(foutput, flabels ,args)
    # take a training step with the discriminator
    G.train()
    G_loss.backward()
    log_batch['G-loss'][batch_idx] = G_loss.item()
    return G_loss