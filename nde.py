import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from nde_models import *

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_nde(model, dataloader, optimizer, epochs=1000, save_dir="checkpoints"):
    model.train()
    os.makedirs(save_dir, exist_ok=True)  # Create directory to save checkpoints
    loss_values = []  # List to store loss values

    for epoch in range(epochs):
        epoch_loss = 0.0
        for theta, x in dataloader:
            # Move data to device
            theta = theta.to(device)
            x = x.to(device)

            # Compute log probability of theta under the flow-based posterior
            log_prob = model(theta, x)  # This now calls the forward method
            loss = -log_prob.mean()  # Negative log-likelihood loss
            epoch_loss += loss.item()
            loss_values.append(loss.item())
            print(loss.item())
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss}")
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # Use model.module to get the underlying model
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
        np.save(os.path.join(save_dir, 'loss_values.npy'), np.array(loss_values))
    # Save the final model and loss values
    torch.save(model.module.state_dict(), os.path.join(save_dir, 'final_model.pt'))  # Use model.module to get the underlying model
    torch.save(optimizer.state_dict(), os.path.join(save_dir, 'final_optimizer.pt'))

# Main
if __name__ == "__main__":
    # Hyperparameters
    theta_dim = 4
    x_dim = 2
    num_samples = 100  # Number of parameter settings
    num_events = 10000    # Number of events per parameter setting
    batch_size =1024
    hidden_dim = 64
    num_layers = 4      # Number of flow layers
    lr = 1e-4
    epochs = 1000
    multi_log = True
    log = False

    # Generate synthetic data
    thetas, xs = generate_data(num_samples, num_events, theta_dim, x_dim, device)

    # Create dataset with pairs of (theta, single_event)
    thetas_repeated = np.repeat(thetas, num_events, axis=0)
    xs_flattened = xs.reshape(-1, x_dim)  # Flatten events into (num_samples * num_events, x_dim)
    thetas_repeated = torch.tensor(thetas_repeated, dtype=torch.float32)
    xs_flattened = torch.tensor(xs_flattened, dtype=torch.float32)
    if multi_log:
        xs_flattened = torch.cat([torch.log(xs_flattened + 1e-8), torch.log10(xs_flattened + 1e-8)], dim=-1).float()
    elif log:
        xs_flattened = torch.log(xs_flattened + 1e-8)

    # Convert to PyTorch tensors
    dataset = TensorDataset(thetas_repeated, xs_flattened)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    theta_ranges = [(5,10), (0.1, 5), (0,2), (7, 9)]  # Define ranges
    theta_ranges_tensor = torch.tensor(theta_ranges, dtype=torch.float32)
    # Initialize model and optimizer
    if multi_log:
        model = NeuralDensityEstimator(theta_dim, 2*x_dim, num_layers, hidden_dim)
    elif log:
        model = NeuralDensityEstimator(theta_dim, x_dim, num_layers, hidden_dim)

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_nde(model, dataloader, optimizer, epochs)