import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from params_learning import LatentToParamsNN, PointNetEmbedding, EventDataset, generate_data, MDN
from simulator import *
from models import *


def up(x, params):
    """Calculate the 'up' function for given x and parameters."""
    return (x ** params[0]) * ((1 - x) ** params[1])


def down(x, params):
    """Calculate the 'down' function for given x and parameters."""
    return (x ** params[2]) * ((1 - x) ** params[3])


def advanced_feature_engineering(xs_tensor):
    """
    Generate advanced features including log, symlog, ratio, and difference.
    
    Args:
        xs_tensor: Tensor with input data.
    
    Returns:
        A tensor with concatenated features.
    """
    log_features = torch.log1p(xs_tensor)
    symlog_features = torch.sign(xs_tensor) * torch.log1p(xs_tensor.abs())

    ratio_features = []
    diff_features = []
    for i in range(xs_tensor.shape[-1]):
        for j in range(i + 1, xs_tensor.shape[-1]):
            ratio = xs_tensor[..., i] / (xs_tensor[..., j] + 1e-8)
            ratio_features.append(torch.log1p(ratio.abs()).unsqueeze(-1))

            diff = torch.log1p(xs_tensor[..., i]) - torch.log1p(xs_tensor[..., j])
            diff_features.append(diff.unsqueeze(-1))

    ratio_features = torch.cat(ratio_features, dim=-1)
    diff_features = torch.cat(diff_features, dim=-1)

    return torch.cat([log_features, symlog_features, ratio_features, diff_features], dim=-1)


def load_and_plot(loss_history_path):
    """Load and plot the training loss history."""
    loss_history = np.load(loss_history_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b', linewidth=2)
    plt.title('Training Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('losses.png')
    
    return loss_history


def plot_loss(loss_history, smooth_window=5):
    """
    Plots loss history with optional smoothing.
    
    Args:
        loss_history: numpy array of loss values.
        smooth_window: size of moving average window (set to 1 for no smoothing).
    """
    plt.figure(figsize=(12, 7))
    plt.plot(loss_history, 'b', alpha=0.3, label='Raw loss')
    
    if smooth_window > 1:
        smooth_loss = np.convolve(loss_history, np.ones(smooth_window) / smooth_window, mode='valid')
        plt.plot(range(smooth_window-1, len(loss_history)), smooth_loss, 'r', linewidth=2, label=f'Smoothed (window={smooth_window})')
    
    plt.title('Training Loss History', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    min_loss = np.min(loss_history)
    min_epoch = np.argmin(loss_history)
    plt.scatter(min_epoch, min_loss, c='green', s=100, label=f'Min loss: {min_loss:.4f}')
    
    plt.legend()
    plt.show()
    
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Minimum loss: {min_loss:.4f} at epoch {min_epoch}")


def evaluate_and_plot(model, pointnet_model, dataloader, device):
    """Evaluate the model and plot error distribution."""
    model.eval()
    predicted_params_list = []
    true_params_list = []

    model.to(device)
    pointnet_model.to(device)

    with torch.no_grad():
        for latent_embeddings, true_params in dataloader:
            latent_embeddings = latent_embeddings.to(device)
            true_params = true_params.to(device)

            predicted_params = model(latent_embeddings)

            predicted_params_list.append(predicted_params)
            true_params_list.append(true_params)

    predicted_params = torch.cat(predicted_params_list, dim=0).squeeze()
    true_params = torch.cat(true_params_list, dim=0)

    errors = torch.abs(predicted_params - true_params) / torch.abs(true_params)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        ax = axes[i]
        ax.hist(errors[:, i].cpu().numpy(), bins=50, alpha=0.7)
        ax.set_title(f'Parameter {i+1}')
        ax.set_xlabel('Error (|theta - theta*|/|theta*|)')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    plt.savefig('Error_Distribution.png')


def sample_from_mdn(pi, mu, sigma):
    """
    Sample from the mixture of Gaussians.
    
    Args:
        pi: Mixture weights.
        mu: Means of the Gaussians.
        sigma: Standard deviations of the Gaussians.
    
    Returns:
        Sampled parameters.
    """
    batch_size, param_dim, num_components = mu.shape
    sampled_params = []

    for b in range(batch_size):
        params_instance = []
        for p in range(param_dim):
            pi_p = pi[b, p, :]
            mu_p = mu[b, p, :]
            sigma_p = sigma[b, p, :]

            component = torch.multinomial(pi_p, 1)
            sample_val = torch.normal(mu_p[component], sigma_p[component])
            params_instance.append(sample_val.squeeze())

        sampled_params.append(torch.stack(params_instance))

    return torch.stack(sampled_params)


def evaluate_and_plot_samples(model, pointnet_model, dataloader, device, n_mc=10):
    """Evaluate the model and plot Monte Carlo samples for parameter distribution."""
    model.eval()
    predicted_samples_list = []
    true_params_list = []

    with torch.no_grad():
        for latent_embeddings, true_params in dataloader:
            latent_embeddings = latent_embeddings.to(device)
            true_params = true_params.to(device)

            pi, mu, sigma = model(latent_embeddings)

            batch_samples = []
            for _ in range(n_mc):
                sampled = sample_from_mdn(pi, mu, sigma)
                batch_samples.append(sampled.unsqueeze(0))
            batch_samples = torch.cat(batch_samples, dim=0)
            predicted_samples_list.append(batch_samples)
            true_params_list.append(true_params.unsqueeze(0).expand(n_mc, -1, -1))

    predicted_samples = torch.cat(predicted_samples_list, dim=1)
    true_params = torch.cat(true_params_list, dim=1)

    fig, axes = plt.subplots(1, true_params.shape[2], figsize=(20, 5))
    for i in range(true_params.shape[2]):
        ax = axes[i]
        samples = predicted_samples[:, :, i].reshape(-1).cpu().numpy()
        ax.hist(samples, bins=50, alpha=0.7)

        true_value = true_params[0, 0, i].item()
        ax.axvline(true_value, color='r', linestyle='dashed', linewidth=2, label='True Value')
        ax.set_title(f'Parameter {i+1}')
        ax.set_xlabel('Predicted value')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig('Params_Distribution_Samples.png')
    plt.show()


def plot_params_distribution_single(model, pointnet_model, true_params, device, n_mc=100):
    """Plot parameter distribution for a single set of true parameters."""
    model.eval()
    pointnet_model.eval()

    true_params = true_params.to(device)
    simulator = SimplifiedDIS(device)
    num_events = 100000
    xs = simulator.sample(true_params, num_events)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    samples = []
    with torch.no_grad():
        for _ in range(n_mc):
            pi, mu, sigma = model(latent_embedding)
            sampled = sample_from_mdn(pi, mu, sigma)
            samples.append(sampled.squeeze(0))

    samples = torch.stack(samples)
    
    fig, axes = plt.subplots(1, true_params.size(0), figsize=(20, 5))
    true_params_np = true_params.cpu().numpy()
    for i in range(true_params.size(0)):
        ax = axes[i]
        ax.hist(samples[:, i].cpu().squeeze().numpy(), bins=50, alpha=0.7)
        ax.axvline(true_params_np[i], color='r', linestyle='dashed', linewidth=2, label='True Value')
        ax.set_title(f'Parameter {i+1}')
        ax.set_xlabel('Predicted value')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig('Single_Params_Distribution_Samples.png')
    plt.show()


def plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=100):
    """Plot predicted PDF distribution for a single set of true parameters."""
    model.eval()
    pointnet_model.eval()

    true_params = true_params.to(device)
    simulator = SimplifiedDIS(device)
    num_events = 100000
    xs = simulator.sample(true_params, num_events)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    pdf_samples = []
    with torch.no_grad():
        for _ in range(n_mc):
            pi, mu, sigma = model(latent_embedding)
            pdf_samples.append((pi, mu, sigma))

    fig, axes = plt.subplots(1, true_params.size(0), figsize=(20, 5))
    true_params_np = true_params.cpu().numpy()
    for i in range(true_params.size(0)):
        ax = axes[i]
        samples_i = [sample[1][:, i].cpu().numpy() for sample in pdf_samples]
        ax.hist(np.concatenate(samples_i), bins=50, alpha=0.7)
        ax.axvline(true_params_np[i], color='r', linestyle='dashed', linewidth=2, label='True Value')
        ax.set_title(f'Parameter {i+1}')
        ax.set_xlabel('Predicted value')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig('Single_Params_PDF_Distribution.png')
    plt.show()
