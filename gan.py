import argparse

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib
import numpy as np
from models import *
from sampler import *
from generator_plotting import *
import sys

sys.path.append('theory/jamlib/')


import csv
import params as par
import cfg
from alphaS  import ALPHAS
from eweak   import EWEAK
from pdf     import PDF
from mellin  import MELLIN
from idis    import THEORY
from mceg    import MCEG

def min_max_normalize(data, max, min):
    return (data - min) / (max- min)

# Turn interactive mode off
plt.ioff()
# theory_cfg = {
#     "parmin": [ -2,   0, -10, -10,  -1,   0, -10, -10,  -1,   0, -10, -10, -10,
#         -2,   0, -10,  -2,   0, -10,  -2,   0, -10, -10, -10,  -2,   0,
#        -10, -10,  -2,   0, -10,  -2,   0],
#     "parmax":[ 10,  10, -10, -10,  10,  10, -10, -10,  10,  10, -10, -10,  10,
#         10,  10,  10,  10,  10,  10,  10,  10, -10, -10,  10,  10,  10,
#        -10, -10,  10,  10,  10,  10,  10],
# }

theory_cfg = {
    "parmin": [0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
    "parmax": [3.0, 1.0, 5.0, 3.0, 1.0, 5.0],
}


def train_loop(
    data,
    Generator,
    Discriminator,
    SurrogatePhysics,
    G_Optimizer,
    D_Optimizer,
    S_Optimizer,
    scheduler,
    args,
    G_losses,
    D_losses,
    S_losses,
    D_accuracies,
    G_accuracies,
    gen_params_means,
    gen_params_variances,
    device,
    tparams,
    dataset,
    filename
):
    data = data.to(device)
    if len(data.size()) < 3:
        data.unsqueeze(-1)
    G_criterion = torch.nn.MSELoss().to(device)
    criterion = torch.nn.MSELoss().to(device)
    surrogate_criterion = nn.MSELoss().to(device)
    D_Optimizer.zero_grad()
    S_Optimizer.zero_grad()
    for _ in range(args.D_steps):
        z = torch.randn(args.batch_size, args.noise_dim).to(device)
        if torch.isnan(data).any():
            print("NaN detected in input data")
            return
        gen_params = Generator(z)
        gen_data = torch.Tensor([]).to(device)
        for i in range(args.batch_size):
            gen_data_i = generate_synthetic_data(
                gen_params[i, :], args.sample_size, option=args.distribution, device=device
            ).float()

            gen_data = torch.cat((gen_data, gen_data_i.unsqueeze(0)), dim=0).to(device)
        if torch.isnan(gen_data).any():
            print("NaN detected in gen_data")
            gen_data[torch.isnan(gen_data)] = 1e-8  # Replace NaN with 1.0
            print("Replaced NaN values with 1.0")
            return
        gen_data = gen_data
        # check how D does with true data
        tlabels = torch.ones(data.size(0), 1).to(device)
        # data = min_max_normalize(data, data.max(), data.min())
        toutput = Discriminator(torch.squeeze(data))
        # # calculate loss on true events
        tloss = criterion(toutput.squeeze(), tlabels.squeeze())

        # check how D does with fake data generated from fake parameters
        flabels = torch.zeros(gen_data.size(0), 1).to(device)
        foutput = Discriminator(torch.squeeze(gen_data))
        if torch.isnan(foutput).any():
            print("NaN detected in Discriminator output")
            return
        # calculate loss on fake events
        floss = criterion(foutput.squeeze().float(), flabels.squeeze())
        D_loss = tloss + floss
        D_losses.append(D_loss.detach().cpu().item())
        D_loss.backward()
        if torch.isnan(D_loss).any():
            print(f"NaN detected in D_loss")
            return
        D_Optimizer.step()
        for j in range(args.S_steps):
            S_Optimizer.zero_grad()
            z = torch.randn(args.batch_size, args.noise_dim).to(device)
            if torch.isnan(data).any():
                print("NaN detected in input data")
                return
            gen_params = Generator(z)
            gen_data = torch.Tensor([]).to(device)
            for i in range(args.batch_size):
                gen_data_i = generate_synthetic_data(
                    gen_params[i, :], args.sample_size, device, option=args.distribution
                ).float()

                gen_data = torch.cat((gen_data, gen_data_i.unsqueeze(0)), dim=0).to(
                    device
                )
            gen_data = gen_data
            foutput = Discriminator(torch.squeeze(gen_data))
            surrogate_floss = SurrogatePhysics(gen_params)
            S_loss = surrogate_criterion(surrogate_floss.squeeze(), foutput.squeeze())
            S_losses.append(S_loss.detach().cpu().item())
            S_loss.backward()
            torch.nn.utils.clip_grad_norm_(SurrogatePhysics.parameters(), max_norm=1.0)
            S_Optimizer.step()

    for _ in range(args.G_steps):
        z = torch.randn(args.batch_size, args.noise_dim).to(device)
        gen_params = Generator(z)
        G_Optimizer.zero_grad()
        # obtain fake loss
        G_output = SurrogatePhysics(gen_params)
        tlabels = torch.ones(G_output.size(0), 1).to(device)
        real_x, real_y = data[..., 0], data[..., 1]
        gen_x, gen_y = gen_data[..., 0], gen_data[..., 1]

        # Penalize deviations in magnitude
        # magnitude_loss = torch.exp(torch.mean((real_x - gen_x) ** 2 + (real_y - gen_y) ** 2))-1
        G_loss = (
            G_criterion(G_output.squeeze(), tlabels.squeeze())
            + 0.05* torch.sum(torch.relu(gen_params - torch.Tensor(theory_cfg["parmax"]).to(device)))
            / len(gen_params)
            + 0.05 * torch.sum(torch.relu(torch.Tensor(theory_cfg["parmin"]).to(device) - gen_params))
            / len(gen_params)
        )

        print("G_loss")

        print(G_loss)
        # Check generator gradients
        for name, param in Generator.named_parameters():
            print(f"{name} requires_grad: {param.requires_grad}")

        for name, param in SurrogatePhysics.named_parameters():
            print(f"{name} requires_grad: {param.requires_grad}")
        G_losses.append(G_loss.detach().cpu().item())
        G_loss.backward()

        # Check if gradients are not None
        for name, param in Generator.named_parameters():
            if param.grad is None:
                print(f"Gradient for {name} is None")
            else:
                print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
        G_Optimizer.step()
    f_predicted = torch.round(foutput)
    t_predicted = torch.max(toutput)
    D_accuracy = 0.5 * (
        (f_predicted == flabels.squeeze()).sum().item() / flabels.size(0)
    ) + 0.5 * ((t_predicted == tlabels.squeeze()).sum().item() / tlabels.size(0))
    D_accuracies.append(D_accuracy)
    G_accuracy = torch.norm(tparams.to(device) - gen_params.mean(dim=0).to(device)) / torch.norm(tparams.to(device))
    G_accuracies.append(G_accuracy.detach().cpu().item())

    print(f"Discriminator Accuracy: {D_accuracy}")
    print(f"Generator Accuracy: {G_accuracy}")
    print(f"True Parameters: {tparams}")
    print(f"Generated Parameters: {torch.mean(gen_params, dim=0).data}")
    gen_params_mean = gen_params.mean(dim=0).detach().cpu().numpy().tolist()
    gen_params_variance = gen_params.var(dim=0).detach().cpu().numpy().tolist()
    gen_params_means.append(gen_params_mean)
    gen_params_variances.append(gen_params_variance)
    # Save metrics to a file after each epoch
    metrics = {
        "G_losses": G_losses,
        "D_losses": D_losses,
        "S_losses": S_losses,
        "D_accuracies": D_accuracies,
        "G_accuracies": G_accuracies,
        "Means": gen_params_means,
        "Variances": gen_params_variances,
    }
    with open(filename+"training_metrics.json", "w") as f:
        json.dump(metrics, f)

    if args.plot == True:
        fig1 = plt.figure(1)
        plt.clf()
        loglog = False
        if args.distribution == "jlab":
            loglog = True
        gen_data = generate_synthetic_data(
            torch.tensor(gen_params_mean), 1024, device, option=args.distribution
        ).float()
        # gen_data = torch.log(gen_data)
        plot_scatter(
            gen_data.detach().cpu().numpy(),
            fig=fig1,
            plot_Gaussian=False,
            plot_other_data=True,
            loglog=loglog,
            other_data=dataset[0:1024, :].detach().cpu().numpy(),
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename + "scatterplot.png")
        plt.draw()
        plt.pause(0.0001)
        plt.close()
        fig2 = plt.figure(2)
        plt.clf()
        ax = fig2.add_subplot(1, 1, 1)
        ax.plot(G_losses, alpha=0.5, label="G Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.set_title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename + "G_losses.png")
        plt.draw()
        plt.pause(0.0001)
        plt.close()
        fig3 = plt.figure(3)
        plt.clf()
        ax = fig3.add_subplot(1, 1, 1)
        ax.plot(D_losses, alpha=0.5, label="D Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.set_title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename + "D_losses.png")
        plt.draw()
        plt.pause(0.0001)
        plt.close()
        fig4 = plt.figure(4)
        plt.clf()
        ax = fig4.add_subplot(1, 1, 1)
        ax.plot(S_losses, alpha=0.5, label="S Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.set_title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename + "S_losses.png")
        plt.draw()
        plt.pause(0.0001)
        plt.close()
        fig5 = plt.figure(5)
        plt.clf()
        plot_params(gen_params_means, gen_params_variances, tparams, fig=fig5)
        plt.tight_layout()
        plt.savefig(filename + "parameters.png")
        plt.draw()
        plt.pause(0.0001)
        plt.close()
    # torch.save(Generator.state_dict(), str(args.filename) + "model.pt")


def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description="Diffusion Denoising Model for Probability Distribution Parameter Estimation"
    )
    parser.add_argument(
        "--noise-dim", type=int, default=64, help="Noise dimension size (default: 100)"
    )
    parser.add_argument(
        "--distribution",
        default="theory",
        help="True data distribution (default: exp)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1,
        help="input sample size for training (default: 1024)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="input batch size for training (default: 3)",
    )
    parser.add_argument(
        "--plot",
        type=bool,
        default=True,
        help="whether or not to plot data during training (default: False)",
    )
    parser.add_argument(
        "--S-steps",
        type=int,
        default=5,
        help="Number of surrogate physics model updates",
    )
    parser.add_argument(
        "--G-steps",
        type=int,
        default=1,
        help="Number of generator model updates",
    )
    parser.add_argument(
        "--D-steps",
        type=int,
        default=1,
        help="Number of discriminator model updates",
    )

    parser.add_argument(
        "--n-true-events",
        type=int,
        default=10240,
        help="Number of true events for training (default: 10000)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="generator",
        help="Filename for results",
    )
    args = parser.parse_args()

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
        tparams = torch.rand(
            2, 2
        ).ravel()  # torch.abs(torch.cat([torch.randn(5, 2).ravel(), torch.rand(5, 2).ravel()]))
    elif args.distribution == "theory":
        tparams = torch.tensor([0.72916667, 0.25, 0.6, 0.36458333, 0.25, 0.8])
    elif args.distribution == "jlab":
        mellin = MELLIN(npts=8)
        alphaS = ALPHAS()
        pdf = PDF(mellin, alphaS)
        tparams = torch.tensor(pdf.current_par)
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Step 1: Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    param_dims = sum(tensor.numel() for tensor in tparams)
    Generator = ConvolutionalGAN(args.noise_dim, param_dims).to(device)
    Discriminator = MLP().to(device)  # PointDiscriminator(loglog).to(device)
    SurrogatePhysics = SurrogatePhysicsModel(input_dim=param_dims).to(device)

    G_Optimizer = optim.Adam(Generator.parameters(), lr=args.lr, betas=(0.5, 0.9999))
    D_Optimizer = optim.Adam(
        Discriminator.parameters(), lr=10 * args.lr, betas=(0.5, 0.9999)
    )
    S_Optimizer = optim.Adam(
        SurrogatePhysics.parameters(), lr=args.lr, betas=(0.5, 0.9999)
    )

    # Learning Rate Scheduler
    scheduler = StepLR(S_Optimizer, step_size=150, gamma=0.1)

    dataset = (
        generate_synthetic_data(
            tparams, args.n_true_events, device, option=args.distribution
        )
        .squeeze()
        .to(device)
    )
    if len(dataset.size()) < 3:
        dataset.unsqueeze(-1)

    # Initialize losses
    G_losses = []
    D_losses = []
    S_losses = []
    D_accuracies = []
    G_accuracies = []
    gen_params_means = []
    gen_params_variances = []

    shuffle_data = True
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle_data)
    Generator.train()
    Discriminator.train()
    SurrogatePhysics.train()
    filename = 'lr' + str(args.lr) + 'noisedim' + str(args.noise_dim) + 'G' + str(args.G_steps) + 'D' + str(args.D_steps) + 'S' + str(args.S_steps) + 'n' + str(args.n_true_events)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for data in data_loader:
            data = data.to(device)
            train_loop(
                data,
                Generator,
                Discriminator,
                SurrogatePhysics,
                G_Optimizer,
                D_Optimizer,
                S_Optimizer,
                scheduler,
                args,
                G_losses,
                D_losses,
                S_losses,
                D_accuracies,
                G_accuracies,
                gen_params_means,
                gen_params_variances,
                device,
                tparams,
                dataset,
                filename
            )

    # torch.save(Generator.state_dict(), str(args.filename) + "model.pt")


if __name__ == "__main__":
    main()
