import argparse

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from models import *
from sampler import *
from plotting import *
from gan_functions import *
import sys
import concurrent
import torch.optim as optim
sys.path.append("theory/jamlib/")

from alphaS import ALPHAS
from pdf import PDF
from mellin import MELLIN

# Turn interactive mode off

plt.ioff()

def train_loop(
    data,
    Generator,
    Discriminator_p,
    Discriminator_n,
    SurrogatePhysics,
    G_Optimizer,
    D_p_Optimizer,
    D_n_Optimizer,
    S_Optimizer,
    args,
    G_losses,
    D_losses,
    S_losses,
    D_accuracies,
    G_accuracies,
    gen_params_means,
    gen_params_quantile_1s,
    gen_params_quantile_3s,
    gen_params_variances,
    device,
    tparams,
    dataset,
    filename,
):
    data = data.to(device)
    if len(data.size()) < 3:
        data.unsqueeze(-1)
    data_p = data[...,0]
    data_n = data[...,1]
    D_criterion = torch.nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss().to(device)
    surrogate_criterion = nn.MSELoss().to(device)
    D_p_Optimizer.zero_grad()
    D_n_Optimizer.zero_grad()
    S_Optimizer.zero_grad()
    for _ in range(args.D_steps):
        z = torch.randn(1, args.noise_dim).to(device)
        if torch.isnan(data).any():
            print("NaN detected in input data")
            return
        gen_params = Generator(z)
        gen_data = torch.Tensor([]).to(device)
        def variance_penalty(points):
            variance = torch.var(points)
            return 1 / (variance + 1e-6)  # add small constant for numerical stability
        def process_data(gen_params_i):
            gen_data_i = generate_synthetic_data(
            gen_params_i,
            args.batch_size,
            option=args.distribution,
            device=device,
            ).float().unsqueeze(0)
            return gen_data_i
        gen_params_list = [gen_params[i, :] for i in np.arange(0,gen_params.size(0)).tolist()]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the function to the data_list, distributing work across threads
            gen_data = list(executor.map(process_data, gen_params_list))
        gen_data = torch.cat(gen_data, dim=0)
        if torch.isnan(gen_data).any():
            print("NaN detected in gen_data")
            gen_data[torch.isnan(gen_data)] = 1e-8  # Replace NaN with 1.0
            print("Replaced NaN values with 1.0")
            return
        gen_data_p = gen_data[...,0]
        gen_data_n = gen_data[...,1]
        print("DATA SIZE")
        print(data.size())
        toutput = Discriminator_p(torch.squeeze(data_p).unsqueeze(-1))
        tlabels = torch.ones(args.batch_size,1).to(device)
        tloss = criterion(toutput.squeeze(), tlabels.squeeze())
        # + torch.log(variance_penalty_non_outliers(data_p))
        D_loss = tloss #+ + 0.01*torch.norm(torch.log(torch.std(data_p)+1e-8)-torch.log(torch.std(gen_data_p)+1e-8))
        # D_losses.append(D_loss.detach().cpu().item())
        # D_loss.backward()
        # torch.nn.utils.clip_grad_norm_(Discriminator_p.parameters(), max_norm=1.0)
        # if torch.isnan(D_loss).any():
        #     print(f"NaN detected in D_loss")
        #     return
        # D_p_Optimizer.step()
        # D_p_Optimizer.zero_grad()

        flabels = torch.zeros(args.batch_size).to(device)

        foutput = Discriminator_p(torch.squeeze(gen_data_p).unsqueeze(-1))
        if torch.isnan(foutput).any() or torch.isinf(foutput).any():
            continue
        floss = criterion(foutput.squeeze().float(), flabels.squeeze()) #+ 0.1*torch.norm(torch.var(gen_data_p)-torch.var(data_p))**2
        D_loss = tloss+floss #+ 0.01*torch.norm(torch.log(torch.std(data_p)+1e-8)-torch.log(torch.std(gen_data_p)+1e-8))
        if torch.isnan(floss).any() or torch.isinf(floss).any():
            continue
        D_losses.append(D_loss.detach().cpu().item())
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(Discriminator_p.parameters(), max_norm=1.0)

        if torch.isnan(foutput).any():
            print("NaN detected in Discriminator output")
            return
        print(foutput.size())

        if torch.isnan(D_loss).any():
            print(f"NaN detected in D_loss")
            return

        D_p_Optimizer.step()

        toutput = Discriminator_n(torch.squeeze(data_n).unsqueeze(-1))

        tloss = criterion(toutput.squeeze(), tlabels.squeeze())
        D_loss = tloss #+ 0.01*torch.norm(torch.log(torch.std(data_n)+1e-8)-torch.log(torch.std(gen_data_n)+1e-8))
        # D_losses.append(D_loss.detach().cpu().item())
        # D_loss.backward()
        # torch.nn.utils.clip_grad_norm_(Discriminator_n.parameters(), max_norm=1.0)
        # if torch.isnan(D_loss).any():
        #     print(f"NaN detected in D_loss")
        #     return
        # D_n_Optimizer.step()
        # D_n_Optimizer.zero_grad()

        foutput = Discriminator_n(torch.squeeze(gen_data_n).unsqueeze(-1))
        if torch.isnan(foutput).any() or torch.isinf(foutput).any():
            continue
        floss = criterion(foutput.squeeze().float(), flabels.squeeze())# + 0.1*torch.norm(torch.var(gen_data_n)-torch.var(data_n))**2
        # if torch.std(gen_data_n) < 0.1:
        #    D_loss += 0.01*torch.norm(torch.log(torch.std(data_n)+1e-8)-torch.log(torch.std(gen_data_n)+1e-8))
        D_loss = tloss+floss
        D_losses.append(D_loss.detach().cpu().item())
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(Discriminator_n.parameters(), max_norm=1.0)
        if torch.isnan(foutput).any():
            print("NaN detected in Discriminator output")
            return
        print(foutput.size())

        if torch.isnan(D_loss).any():
            print(f"NaN detected in D_loss")
            return
        # torch.nn.utils.clip_grad_norm_(Discriminator_n.parameters(), max_norm=0.5)

        D_n_Optimizer.step()

        for _ in range(args.S_steps):
            Discriminator_p.eval()
            Discriminator_n.eval()
            Generator.eval()
            z = torch.randn(1, args.noise_dim).to(device)
            gen_params = Generator(z)
            gen_data = torch.Tensor([]).to(device)
            def process_data(gen_params_i):
                gen_data_i = generate_synthetic_data(
                gen_params_i,
                args.batch_size,
                option=args.distribution,
                device=device,
                ).float().unsqueeze(0)
                return gen_data_i
            # breakpoint()
            gen_params_list = [gen_params[i, :] for i in np.arange(0,gen_params.size(0)).tolist()]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Map the function to the data_list, distributing work across threads
                gen_data = list(executor.map(process_data, gen_params_list))
            gen_data = torch.cat(gen_data, dim=0)
            gen_data_p = gen_data[...,0]
            gen_data_n = gen_data[...,1]
            foutput_p = Discriminator_p(torch.squeeze(gen_data_p).unsqueeze(-1))
            foutput_n = Discriminator_n(torch.squeeze(gen_data_n).unsqueeze(-1))
            # breakpoint()
            foutput = torch.mean(torch.cat((foutput_p, foutput_n),dim=-1), dim = -1)

            surrogate_floss = SurrogatePhysics(gen_params.repeat(args.batch_size,1))
            if torch.isnan(surrogate_floss).any() or torch.isinf(surrogate_floss).any():
                print("surrogate_floss_p Loss contains NaN or Inf values!")
            # foutput_p = Discriminator_p(torch.squeeze(gen_data_p).unsqueeze(-1))
            if torch.isnan(foutput_p).any() or torch.isinf(foutput_p).any() or torch.isnan(foutput_n).any() or torch.isinf(foutput_n).any():
                continue
            S_loss = surrogate_criterion(
                surrogate_floss.squeeze(), foutput.squeeze().detach()
            )
            S_loss.backward()
            S_losses.append(S_loss.detach().cpu().item())
            # torch.nn.utils.clip_grad_norm_(SurrogatePhysics.parameters(), max_norm=0.5)
            S_Optimizer.step()
            S_Optimizer.zero_grad()
            if torch.isnan(S_loss).any() or torch.isinf(S_loss).any():
                print("Loss contains NaN or Inf values!")
                S_loss = torch.where(torch.isnan(S_loss), torch.tensor(0.0, device=S_loss.device), S_loss)
                S_loss = torch.where(torch.isinf(S_loss), torch.tensor(0.0, device=S_loss.device), S_loss)
                continue
            # surrogate_floss_n = SurrogatePhysics(gen_params_n.repeat(args.batch_size,1))

            # foutput_n = Discriminator_n(torch.squeeze(gen_data_n).unsqueeze(-1))
            # S_loss = surrogate_criterion(
            #     surrogate_floss_n.squeeze(), foutput_n.squeeze().detach()
            # )
            # S_loss.backward()
            # S_losses.append(S_loss.detach().cpu().item())
            # # torch.nn.utils.clip_grad_norm_(SurrogatePhysics.parameters(), max_norm=1.0)
            # S_Optimizer.step()

    for _ in range(args.G_steps):
        Generator.train()
        z = torch.randn(args.batch_size, args.noise_dim).to(device)
        gen_params = Generator(z)
        G_Optimizer.zero_grad()
        # obtain fake loss

        gen_data = torch.Tensor([]).to(device)
        def process_data(gen_params_i):
            gen_data_i = generate_synthetic_data(
            gen_params_i,
            args.batch_size,
            option=args.distribution,
            device=device,
            ).float().unsqueeze(0)
            return gen_data_i
        gen_params_list = [gen_params[i, :] for i in np.arange(0,gen_params.size(0)).tolist()]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the function to the data_list, distributing work across threads
            gen_data = list(executor.map(process_data, gen_params_list))
        gen_data = torch.cat(gen_data, dim=0)
        if torch.isnan(gen_data).any():
            print("NaN detected in gen_data")
            gen_data[torch.isnan(gen_data)] = 1e-8  # Replace NaN with 1.0
            print("Replaced NaN values with 1.0")
            return
        gen_data_p = gen_data[...,0]
        gen_data_n = gen_data[...,1]

        G_output = SurrogatePhysics(gen_params)
        if torch.isnan(G_output).any() or torch.isinf(G_output).any():
            continue
        tlabels = torch.ones(args.batch_size, 1).to(device)
        G_loss = criterion(G_output.squeeze(), tlabels.squeeze()) 
        # G_loss = generator_hinge_loss(G_output.squeeze()).to(device)

        print("G_loss")
        print(G_loss)
        # Check generator gradients
        for name, param in Generator.named_parameters():
            print(f"{name} requires_grad: {param.requires_grad}")

        for name, param in SurrogatePhysics.named_parameters():
            print(f"{name} requires_grad: {param.requires_grad}")
        G_losses.append(G_loss.detach().cpu().item())
        G_loss.backward()
        print("GENERATOR")
        # Check if gradients are not None
        for name, param in Generator.named_parameters():
            if param.grad is None:
                print(f"Gradient for {name} is None")
            else:
                print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
        print("DISCRIMINATOR p")  
        # Check if gradients are not None
        for name, param in Discriminator_p.named_parameters():
            if param.grad is None:
                print(f"Gradient for {name} is None")
            else:
                print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
        print("DISCRIMINATOR n")  
        # Check if gradients are not None
        for name, param in Discriminator_n.named_parameters():
            if param.grad is None:
                print(f"Gradient for {name} is None")
            else:
                print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
        print("SURROGATE PHYSICS")
        # Check if gradients are not None
        for name, param in SurrogatePhysics.named_parameters():
            if param.grad is None:
                print(f"Gradient for {name} is None")
            else:
                print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
        G_Optimizer.step()
    Discriminator_p.train()
    Discriminator_n.train()
    f_predicted = torch.round(foutput)
    t_predicted = torch.max(toutput)
    D_accuracy = 0.5 * (
        (f_predicted == flabels.squeeze()).sum().item() / flabels.size(0)
    ) + 0.5 * ((t_predicted == tlabels.squeeze()).sum().item() / tlabels.size(0))
    D_accuracies.append(D_accuracy)
    G_accuracy = torch.norm(
        tparams.to(device) - gen_params.mean(dim=0).to(device)
    ) / torch.norm(tparams.to(device))
    G_accuracies.append(G_accuracy.detach().cpu().item())

    print(f"Discriminator Accuracy: {D_accuracy}")
    print(f"Generator Accuracy: {G_accuracy}")
    print(f"True Parameters: {tparams}")
    print(f"Generated Parameters: {torch.mean(gen_params, dim=0).data}")
    gen_params_mean = torch.quantile(gen_params,0.5,dim=0).detach().cpu().numpy().tolist()
    gen_params_quantile_1 =torch.quantile(gen_params, 0.25, dim=0).detach().cpu().numpy().tolist()
    gen_params_quantile_3 =torch.quantile(gen_params, 0.75, dim=0).detach().cpu().numpy().tolist()
    gen_params_means.append(gen_params_mean)
    gen_params_variances.append(gen_params.var(dim=0).detach().cpu().numpy().tolist())
    gen_params_quantile_1s.append(gen_params_quantile_1)
    gen_params_quantile_3s.append(gen_params_quantile_3)
    # Save metrics to a file after each epoch
    metrics = {
        "G_losses": G_losses,
        "D_losses": D_losses,
        "S_losses": S_losses,
        "D_accuracies": D_accuracies,
        "G_accuracies": G_accuracies,
        "Medians": gen_params_means,
        "25": gen_params_quantile_1s,
        "75": gen_params_quantile_3s
    }
    with open('results/' + filename + "training_metrics_3_zero_optimizer.json", "w") as f:
        json.dump(metrics, f)

    if args.plot == True:
        fig1 = plt.figure(1)
        plt.clf()
        loglog = False
        if args.distribution == "jlab":
            loglog = True
        gen_data = generate_synthetic_data(
            torch.tensor(gen_params.mean(dim=0).detach().cpu().numpy().tolist()).to(device), 1024, device, option=args.distribution
        ).float()
        # Select 1024 random indices
        random_indices = np.random.choice(args.n_true_events, size=1024, replace=False)
        if gen_params.mean(dim=0).size(0) > 1:
            plot_scatter(
                gen_data.detach().cpu().numpy(),
                fig=fig1,
                plot_Gaussian=False,
                plot_other_data=True,
                loglog=loglog,
                other_data=dataset[random_indices].detach().cpu().numpy(),
            )
        else:
            plt.hist(gen_data.squeeze().detach().cpu().numpy(), bins = 100, alpha = 0.5, label = "Generated Data")
            plt.hist(dataset[random_indices].squeeze().detach().cpu().numpy(), bins = 100, alpha = 0.5, label = "True Data")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/" + filename + "scatterplot_3_zero_optimizer.png")
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
        plt.savefig("results/" + filename + "G_losses_3_zero_optimizer.png")
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
        ax.set_yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/" + filename + "D_losses_3_zero_optimizer.png")
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
        plt.savefig("results/" + filename + "S_losses_3_zero_optimizer.png")
        plt.draw()
        plt.pause(0.0001)
        plt.close()
        fig5 = plt.figure(5)
        plt.clf()
        plot_params(gen_params_means, gen_params_variances, tparams, fig=fig5)
        plt.tight_layout()
        plt.savefig("results/" + filename + "parameters_3_zero_optimizer.png")
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
        "--noise-dim", type=int, default=128, help="Noise dimension size (default: 100)"
    )
    parser.add_argument(
        "--distribution",
        default="SimplifiedDIS",
        help="True data distribution (default: exp)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42 )"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        default=128,
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
        default=8,
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
        default=3,
        help="Number of discriminator model updates",
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

    # Set true parameters
    if args.distribution == "poisson":
        tparams = torch.tensor([7.0])
    elif args.distribution == "SimplifiedDIS":
        tparams = torch.tensor([ -0.5, 3,  -0.5, 4])
    elif args.distribution == "mixture":
        tparams = torch.tensor([2.0,2.0,5.0,5.0]) # torch.abs(torch.cat([torch.randn(5, 2).ravel(), torch.rand(5, 2).ravel()]))
    elif args.distribution == "jlab":
        mellin = MELLIN(npts=8)
        alphaS = ALPHAS()
        pdf = PDF(mellin, alphaS)
        tparams = torch.tensor(pdf.current_par)[4:8]
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Step 1: Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    param_dims = sum(tensor.numel() for tensor in tparams)
    Generator = ConvolutionalGAN(args.noise_dim, param_dims).to(device)
    Discriminator_p = MLP(input_dim = 1).to(device)
    Discriminator_n = MLP(input_dim = 1).to(device)
    SurrogatePhysics = SurrogatePhysicsModel(input_dim=param_dims).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        Generator = torch.nn.DataParallel(Generator).to(device)
        Discriminator_p = torch.nn.DataParallel(Discriminator_p).to(device)
        Discriminator_n = torch.nn.DataParallel(Discriminator_n).to(device)
        SurrogatePhysics = torch.nn.DataParallel(SurrogatePhysics).to(device)
    G_Optimizer = optim.Adam(Generator.parameters(), lr=args.lr, betas = (0.5,0.9999))
    D_p_Optimizer = optim.Adam(Discriminator_p.parameters(), lr=args.lr, betas = (0.5,0.9999))
    D_n_Optimizer = optim.Adam(Discriminator_n.parameters(), lr=args.lr, betas = (0.5,0.9999))
    S_Optimizer = optim.Adam(SurrogatePhysics.parameters(), lr=0.1*args.lr, betas = (0.5,0.9999))

    print(f"Number of trainable parameters in G: {sum(p.numel() for p in Generator.parameters() if p.requires_grad)}")
    print(f"Number of trainable parameters in D_p: {sum(p.numel() for p in Discriminator_p.parameters() if p.requires_grad)}")
    print(f"Number of trainable parameters in D_n: {sum(p.numel() for p in Discriminator_n.parameters() if p.requires_grad)}")
    print(f"Number of trainable parameters in S: {sum(p.numel() for p in SurrogatePhysics.parameters() if p.requires_grad)}")

    dataset = (
        generate_synthetic_data(
            tparams, args.n_true_events, device, option=args.distribution
        )
        .squeeze()
        .to(device)
    )
    if len(dataset.size()) < 3:
        dataset.unsqueeze(-1)

    
    # class MyDataset(Dataset):
    #     def __init__(self, data, labels, outlier_indices):
    #         self.data = data
    #         self.labels = labels
    #         self.outlier_indices = outlier_indices
        
    #     def __len__(self):
    #         return len(self.data)
        
    #     def __getitem__(self, idx):
    #         return self.data[idx], self.labels[idx]


    # Initialize losses
    G_losses = []
    D_losses = []
    S_losses = []
    D_accuracies = []
    G_accuracies = []
    gen_params_means = []
    gen_params_quantile_1s = []
    gen_params_quantile_3s = []
    gen_params_variances = []

    shuffle_data = True
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=shuffle_data
    )
    Generator.train()
    Discriminator_p.train()
    Discriminator_n.train()
    SurrogatePhysics.train()
    filename = (
        "lr"
        + str(args.lr)
        + "noisedim"
        + str(args.noise_dim)
        + "G"
        + str(args.G_steps)
        + "D"
        + str(args.D_steps)
        + "S"
        + str(args.S_steps)
        + "n"
        + str(args.n_true_events)
        + "seed"
        + str(args.seed)
        + "batchsize"
        + str(args.batch_size)
        + args.distribution
        + "hinge"
    )

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for data in data_loader:
            data = data.to(device)
            train_loop(
                data,
                Generator,
                Discriminator_p,
                Discriminator_n,
                SurrogatePhysics,
                G_Optimizer,
                D_p_Optimizer,
                D_n_Optimizer,
                S_Optimizer,
                args,
                G_losses,
                D_losses,
                S_losses,
                D_accuracies,
                G_accuracies,
                gen_params_means,
                gen_params_quantile_1s,
                gen_params_quantile_3s,
                gen_params_variances,
                device,
                tparams,
                dataset,
                filename,
            )

            torch.save(Generator.state_dict(), filename + "model_3_zero_optimizer.pt")
            if len(G_losses) > 1500:
                break
        if len(G_losses) > 1500:
            break


if __name__ == "__main__":
    main()
