import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
from nde_models import *

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the checkpointed model
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
    return model, optimizer, epoch, loss

# Generate data from the simulator for ONE parameter setting
def generate_data_for_one_parameter(true_theta, num_events, device=torch.device("cpu")):
    """
    Generate data for ONE parameter setting using the simulator.
    """
    simulator = SimplifiedDIS(device)
    xs = simulator.sample(true_theta, num_events).cpu().numpy()
    
    # Convert to PyTorch tensors and move to the correct device
    xs = torch.tensor(xs, dtype=torch.float32).to(device)
    
    # Apply log transformation if needed
    if multi_log:
        xs = torch.cat([torch.log(xs + 1e-8), torch.log10(xs + 1e-8)], dim=-1).float()
    elif log:
        xs = torch.log(xs + 1e-8)
    
    return xs

# Estimate parameters for each event using the NDE model
def estimate_parameters_for_events(model, xs, num_samples=1000):
    """
    Estimate parameters for each event using the NDE model.
    """
    model.eval()
    with torch.no_grad():
        # Generate theta_samples from the prior (e.g., uniform distribution)
        theta_samples = np.random.uniform(low=0, high=5.0, size=(num_samples, model.theta_dim))
        theta_samples = torch.tensor(theta_samples, dtype=torch.float32).to(device)
        
        # Repeat xs to match the number of theta samples
        xs_repeated = xs.unsqueeze(0).repeat(num_samples, 1, 1)  # Shape: (num_samples, num_events, x_dim)
        xs_repeated = xs_repeated.view(-1, xs.shape[1])  # Flatten to (num_samples * num_events, x_dim)
        
        # Repeat theta_samples to match the number of events
        theta_samples_repeated = theta_samples.unsqueeze(1).repeat(1, xs.shape[0], 1)  # Shape: (num_samples, num_events, theta_dim)
        theta_samples_repeated = theta_samples_repeated.view(-1, model.theta_dim)  # Flatten to (num_samples * num_events, theta_dim)
        
        # Compute log probabilities for each theta sample and event
        log_probs = model.log_prob(theta_samples_repeated, xs_repeated)
        
        # Reshape log_probs to (num_samples, num_events)
        log_probs = log_probs.view(num_samples, xs.shape[0])
        
        # Find the theta sample with the highest log probability for each event
        best_theta_indices = torch.argmax(log_probs, dim=0)  # Shape: (num_events,)
        best_thetas = theta_samples[best_theta_indices]  # Shape: (num_events, theta_dim)
    
    return best_thetas

# Plot the estimated distribution of each parameter in a 2x2 subfigure layout
def plot_parameter_distribution(best_thetas, true_theta):
    """
    Plot the estimated distribution of each parameter with a vertical line for the true value.
    Uses a 2x2 subfigure layout.
    """
    best_thetas = best_thetas.cpu().numpy()
    num_params = best_thetas.shape[1]
    
    # Create a 2x2 subfigure layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()  # Flatten the axes array for easy indexing
    
    # Plot each parameter in its own subplot
    for i in range(num_params):
        ax = axes[i]
        ax.hist(best_thetas[:, i], bins=20, density=True, alpha=0.6, color='blue', label=f'Parameter {i+1}')
        ax.axvline(true_theta[i], color='red', linestyle='--', label=f'True Parameter {i+1}')
        ax.set_xlabel(f'Parameter {i+1} Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    # Remove empty subplots (if any)
    for i in range(num_params, 4):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    plt.savefig('distributions.png')

# Main
if __name__ == "__main__":
    # Hyperparameters
    theta_dim = 4
    x_dim = 2
    num_events = 10000  # Number of events to generate
    hidden_dim = 64
    num_layers = 4      # Number of flow layers
    lr = 1e-3
    multi_log = True
    log = False

    # True parameter setting (replace with your desired values)
    true_theta = np.array([2.0, 2.0, 2.0, 2.0])  # Example true parameter values

    # Initialize model and optimizer
    if multi_log:
        model = NeuralDensityEstimator(theta_dim, 2*x_dim, num_layers, hidden_dim).to(device)
    elif log:
        model = NeuralDensityEstimator(theta_dim, x_dim, num_layers, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the checkpointed model
    checkpoint_path = "checkpoints/checkpoint_epoch_500.pt"  # Replace with the path to your checkpoint
    model, optimizer, epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)

    # Generate data for ONE parameter setting
    xs = generate_data_for_one_parameter(true_theta, num_events, device)

    # Estimate parameters for each event using the NDE model
    best_thetas = estimate_parameters_for_events(model, xs)

    # Plot the estimated distribution of each parameter
    plot_parameter_distribution(best_thetas, true_theta)