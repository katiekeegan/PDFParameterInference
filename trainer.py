import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from simulator import *
from models import *
from dataloader import *
class DDPMLoss:
    def __call__(self, pred_noise, true_noise):
        # Compute the mean squared error between predicted noise and true noise
        return torch.mean((pred_noise - true_noise) ** 2)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    """Linear noise schedule for the diffusion process."""
    return torch.linspace(beta_start, beta_end, timesteps)

def get_noisy_params(params, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """Add noise to the parameters according to the diffusion process."""
    sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    noise = torch.randn_like(params)
    noisy_params = sqrt_alpha_t * params + sqrt_one_minus_alpha_t * noise
    return noisy_params, noise

def train_diffusion_model(param_ranges, sample_dim,  nevents, param_dim, 
                          hidden_dim=128, timesteps=1000, epochs=1000, lr=1e-3, batch_size=32, total_dataset_size=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UniformDataset(param_ranges, nevents, device=device, total_dataset_size=total_dataset_size)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DiffusionModel(sample_dim=sample_dim, param_dim=param_dim, hidden_dim=hidden_dim, timesteps=timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DDPMLoss()

    # Define the noise schedule
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        for samples, params in dataloader:
            params, samples = params.to(device), samples.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (params.size(0),), device=device)
            
            # Add noise to the parameters according to the schedule
            noisy_params, noise = get_noisy_params(params, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            
            # Predict the noise and compute the loss
            optimizer.zero_grad()
            pred_noise = model(samples, t, noisy_params)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
    
    print("Training complete.")
    return model

if __name__ == "__main__":
    param_ranges = torch.tensor([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    nevents = 10000
    batch_size = 16
    sample_dim = 2
    param_dim = 4
    hidden_dim = 32
    epochs = 100000
    lr = 1e-3
    timesteps = 100
    total_dataset_size = 1000
    
    trained_model = train_diffusion_model(param_ranges,sample_dim, nevents, param_dim, hidden_dim, timesteps, epochs, lr, batch_size, total_dataset_size)
    print("Training finished.")