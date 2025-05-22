import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PDF_learning import *
from simulator import *
from models import *
from torch.distributions import *
import os

def plot_loss_curves(loss_dir='.', save_path='loss_plot.png', show_plot=True):
    """
    Plots training loss components from .npy files in the given directory.

    Args:
        loss_dir (str): Path to the directory containing the loss .npy files.
        save_path (str): Path to save the output plot image.
        show_plot (bool): Whether to display the plot interactively.
    """
    # Build full paths
    # total_path = os.path.join(loss_dir, 'loss_total.npy')
    contrastive_path = os.path.join(loss_dir, 'loss_contrastive.npy')
    regression_path = os.path.join(loss_dir, 'loss_regression.npy')

    # Load data
    contrastive_loss = np.load(contrastive_path)
    regression_loss = np.load(regression_path)
    # breakpoint()

    # Plotting
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(contrastive_loss) + 1)

    # plt.plot(epochs, total_loss, label='Total Loss', linewidth=2)
    plt.plot(epochs, contrastive_loss, label='Contrastive Loss', linewidth=2)
    plt.plot(epochs, regression_loss, label='Regression Loss (scaled)', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    total_path = os.path.join(loss_dir, 'loss_total.npy')
    total_loss = np.load(total_path)
    # Plotting
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(total_loss) + 1)

    plt.plot(epochs, total_loss, label='Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('loss_PDF_learning.png', dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(total_loss) + 1)

    plt.plot(epochs, total_loss, label='Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Training Loss Components Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('log_loss_PDF_learning.png', dpi=300)
    if show_plot:
        plt.show()
    plt.close()




def compute_chisq_statistic(true_function, predicted_function):
    """
    Computes the Chi-square statistic between the true function and the predicted function.
    
    Args:
        true_function: The true function values (observed data).
        predicted_function: The predicted function values (expected data).
        
    Returns:
        The Chi-square statistic.
    """
    # Compute Chi-square statistic
    chisq = np.sum(((true_function - predicted_function) ** 2) / (predicted_function + 1e-10))  # Adding small value to avoid division by zero
    return chisq

def evaluate_over_n_samples(model, pointnet_model, n=100, num_events=100000, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = SimplifiedDIS(torch.device('cpu'))

    all_errors = []
    chi2_up = []
    chi2_down = []

    for i in range(n):
        # === 1. Sample true parameters from a known range (e.g., Uniform[1, 10]) ===
        true_params = torch.FloatTensor(4).uniform_(0.0, 5.0).to(device)

        # === 2. Generate data from simulator ===
        xs = simulator.sample(true_params.cpu(), num_events).to(device)
        xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_feat = advanced_feature_engineering(xs_tensor)
        latent = pointnet_model(xs_feat.unsqueeze(0))

        # === 3. Sample predicted parameters ===
        # breakpoint()
        samples = sample_with_mc_dropout(model, torch.tensor(latent).unsqueeze(0), 1000).squeeze()  # (1000, 4)
        predicted = torch.mean(samples, dim=0)

        # === 4. Compute relative error ===
        error = torch.abs(predicted - true_params) / (true_params + 1e-8)
        all_errors.append(error.cpu())

        # === 5. Evaluate Chi-squared statistics ===
        x_grid = torch.linspace(0.01, 0.99, 1000).to(device)  # Avoid 0/1
        pred_up = torch.mean(torch.stack([up(x_grid.cpu(), p.cpu()) for p in samples]), dim=0)
        pred_down = torch.mean(torch.stack([down(x_grid.cpu(), p.cpu()) for p in samples]), dim=0)
        true_up = up(x_grid, true_params)
        true_down = down(x_grid, true_params)

        chi2_up.append(compute_chisq_statistic(true_up.cpu().numpy(), pred_up.cpu().numpy()))
        chi2_down.append(compute_chisq_statistic(true_down.cpu().numpy(), pred_down.cpu().numpy()))

    all_errors = torch.stack(all_errors).numpy()

    # === 6. Plot parameter-wise error distribution ===
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axes[i].hist(all_errors[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Parameter {i+1} Relative Error')
        axes[i].set_xlabel('|θ_pred - θ_true| / |θ_true|')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("error_distributions.png")
    plt.show()

    # === 7. Print and plot chi-squared ===
    chi2_up = np.array(chi2_up)
    chi2_down = np.array(chi2_down)
    print(f"Mean Chi² up: {chi2_up.mean():.4f} ± {chi2_up.std():.4f}")
    print(f"Mean Chi² down: {chi2_down.mean():.4f} ± {chi2_down.std():.4f}")

    plt.figure(figsize=(10, 5))
    plt.hist(chi2_up, bins=50, alpha=0.6, label='Chi² Up')
    plt.hist(chi2_down, bins=50, alpha=0.6, label='Chi² Down')
    plt.legend()
    plt.title("Chi² Statistic Distribution")
    plt.xlabel("Chi²")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("chisq_distributions.png")
    plt.show()

def enable_dropout(model):
    """Enable dropout layers during evaluation"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def sample_with_mc_dropout(model, latent_embedding, n_samples=100):
    """Draw samples with Monte Carlo dropout"""
    enable_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(latent_embedding)
            samples.append(output)
    return torch.stack(samples)


def plot_2d_histograms(true_params, generated_params, simulator, num_events=100000, bins=50, device=None):
    true_events = simulator.sample(true_params, num_events).cpu().numpy()
    generated_events = simulator.sample(generated_params, num_events).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].scatter(true_events[:, 0], true_events[:, 1],cmap='Blues')
    axes[0].set_title("2D Histogram - True Parameters")
    axes[0].set_xlabel("Sigma_p")
    axes[0].set_ylabel("Sigma_n")

    axes[1].scatter(generated_events[:, 0], generated_events[:, 1], cmap='Oranges')
    axes[1].set_title("2D Histogram - Generated Parameters")
    axes[1].set_xlabel("Sigma_p")
    axes[1].set_ylabel("Sigma_n")

    plt.tight_layout()
    plt.savefig("histograms.png")
    plt.show()

def up(x, params):
    return (x ** params[0]) * ((1 - x) ** params[1])

def down(x, params):
    return (x ** params[2]) * ((1 - x) ** params[3])

class SimplifiedDIS:
    def __init__(self, device=None):
        self.device = device

    def init(self, params):
        self.Nu = 1
        self.au = params[0]
        self.bu = params[1]
        self.Nd = 2
        self.ad = params[2]
        self.bd = params[3]

    def up(self, x):
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(torch.tensor(params, device=self.device))

        xs_p = torch.rand(nevents, device=self.device)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)

        xs_n = torch.rand(nevents, device=self.device)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()

def advanced_feature_engineering(xs_tensor):
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

def plot_event_histogram(model, pointnet_model, true_params, device, n_mc=100, num_events=100000):
    model.eval()
    pointnet_model.eval()
    simulator = SimplifiedDIS(torch.device('cpu'))

    # Simulate true events
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.cpu(), num_events).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # Monte Carlo sampling
    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()

    if torch.any(torch.isnan(samples)):
        print("NaNs detected in samples!")
        return

    mode_params = torch.median(samples, dim=0).values  # median for robustness

    if torch.any(torch.isnan(mode_params)):
        print("NaNs detected in mode parameters!")
        return

    generated_events = simulator.sample(mode_params.cpu(), num_events).to(device)
    true_events_np = xs.cpu().numpy()
    generated_events_np = generated_events.cpu().numpy()

    if np.any(np.isnan(generated_events_np)) or np.any(np.isnan(true_events_np)):
        print("NaNs detected in the events!")
        return

    print(f"Shape of true_events_np: {true_events_np.shape}")
    print(f"Shape of generated_events_np: {generated_events_np.shape}")

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].scatter(true_events_np[:, 0], true_events_np[:, 1], cmap='viridis')
    axs[0].set_title('True Events Histogram')
    axs[0].set_xlabel('Event X')
    axs[0].set_ylabel('Event Y')
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    axs[1].scatter(generated_events_np[:, 0], generated_events_np[:, 1],cmap='viridis')
    axs[1].set_title('Generated Events Histogram')
    axs[1].set_xlabel('Event X')
    axs[1].set_ylabel('Event Y')
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("histograms.png")
    plt.show()

def plot_params_distribution_single(model, pointnet_model, true_params, device, n_mc=100):
    model.eval()
    pointnet_model.eval()
    simulator = SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)
    xs = simulator.sample(true_params.cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()

    fig, axes = plt.subplots(1, true_params.size(0), figsize=(20, 4))
    for i in range(true_params.size(0)):
        axes[i].hist(samples[:, i].cpu().numpy(), bins=20, alpha=0.7, color='skyblue')
        axes[i].axvline(true_params[i].item(), color='red', linestyle='dashed', label='True Value')
        axes[i].set_title(f'Param {i+1}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("Dist.png")
    plt.show()

def plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=100):
    model.eval()
    pointnet_model.eval()
    simulator = SimplifiedDIS(torch.device('cpu'))

    # Prepare input
    true_params = true_params.to(device)
    xs = simulator.sample(true_params.cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # Monte Carlo sampling
    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()
    x_vals = torch.linspace(0, 1, 500).to(device)

    # --- Compute up(x) stats ---
    up_vals_all = []
    for i in range(n_mc):
        sample_params = samples[i]
        simulator.init(sample_params)
        up_vals = simulator.up(x_vals)
        up_vals_all.append(up_vals.unsqueeze(0))
    
    # Assume up_vals_stack shape: [n_samples, num_points]
    up_vals_stack = torch.cat(up_vals_all, dim=0)

    # # Remove bad rows (e.g. anything with near-zero values)
    # valid_mask = (up_vals_stack > 1e-16).all(dim=1)
    # if valid_mask.sum() == 0:
    #     print("⚠️ No valid up(x) samples remained. Using all samples.")
    #     valid_mask = torch.ones(up_vals_stack.shape[0], dtype=torch.bool)
    # up_vals_stack = up_vals_stack[valid_mask]

    # Compute median and quantiles
    median_up_vals = up_vals_stack.median(dim=0).values
    lower_up = torch.quantile(up_vals_stack, 0.1, dim=0)
    upper_up = torch.quantile(up_vals_stack, 0.9, dim=0)

    # # Clamp lower bound for log scale
    # lower_up = torch.clamp(lower_up, min=1e-6)

    simulator.init(true_params.squeeze())
    true_up_vals = simulator.up(x_vals)

    # Plot UP
    fig_up, ax_up = plt.subplots(figsize=(8, 6))
    ax_up.plot(x_vals.cpu(), true_up_vals.cpu(), label="True up(x)", color='blue')
    ax_up.plot(x_vals.cpu(), median_up_vals.cpu(), label="Median predicted up(x)", color='red', linestyle='--')
    ax_up.fill_between(
        x_vals.cpu(),
        lower_up.cpu(),
        upper_up.cpu(),
        color='red',
        alpha=0.3,
        label="IQR"
    )
    ax_up.set_title("Comparison of up(x)")
    ax_up.set_xlabel("x")
    ax_up.set_xscale("log")
    ax_up.set_ylabel("up(x)")
    ax_up.set_yscale("log")
    ax_up.set_xlim(0, 1)
    ax_up.legend()
    plt.tight_layout()
    plt.savefig("up.png")
    plt.close(fig_up)


    # --- Compute down(x) stats ---
    down_vals_all = []
    for i in range(n_mc):
        sample_params = samples[i]
        simulator.init(sample_params)
        down_vals = simulator.down(x_vals)
        down_vals_all.append(down_vals.unsqueeze(0))

    # Assume up_vals_stack shape: [n_samples, num_points]
    down_vals_stack = torch.cat(down_vals_all, dim=0)
    # breakpoint()
    # # Remove bad rows (e.g. anything with near-zero values)
    # valid_mask = (down_vals_stack > 1e-16).all(dim=1)
    # if valid_mask.sum() == 0:
    #     print("⚠️ No valid down(x) samples remained. Using all samples.")
    #     valid_mask = torch.ones(down_vals_stack.shape[0], dtype=torch.bool)
    # down_vals_stack = down_vals_stack[valid_mask]

    # Compute median and quantiles
    median_down_vals = down_vals_stack.median(dim=0).values
    lower_down = torch.quantile(down_vals_stack, 0.1, dim=0)
    upper_down = torch.quantile(down_vals_stack, 0.9, dim=0)

    # # Clamp lower bound for log scale
    # lower_down = torch.clamp(lower_down, min=1e-6)

    simulator.init(true_params.squeeze())
    true_down_vals = simulator.down(x_vals)

    # Plot DOWN
    fig_down, ax_down = plt.subplots(figsize=(8, 6))
    ax_down.plot(x_vals.cpu(), true_down_vals.cpu(), label="True down(x)", color='blue')
    ax_down.plot(x_vals.cpu(), median_down_vals.cpu(), label="Median predicted down(x)", color='red', linestyle='--')
    ax_down.fill_between(
        x_vals.cpu(),
        lower_down.cpu(),
        upper_down.cpu(),
        color='red',
        alpha=0.3,
        label="IQR"
    )
    ax_down.set_title("Comparison of down(x)")
    ax_down.set_xlabel("x")
    ax_down.set_xscale("log")
    ax_down.set_ylabel("down(x)")
    ax_down.set_yscale("log")
    ax_down.set_xlim(0, 1)
    ax_down.legend()
    plt.tight_layout()
    plt.savefig("down.png")
    plt.close(fig_down)




# Load the model and data
def load_model_and_data(model_path, pointnet_model_path, num_samples=100, num_events=10000, device=None):
    # Ensure the device is set
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate test data
    thetas, xs = generate_data(num_samples, num_events, device=torch.device('cpu'))
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor_engineered = advanced_feature_engineering(xs)
    input_dim = xs_tensor_engineered.shape[-1]

    # Instantiate the models
    latent_dim = 1024  # Example latent dimension
    # model = ImprovedMDN(latent_dim, 4)  # Parameter dimension is 4
    model = InferenceNet(embedding_dim=latent_dim).to(device)
    pointnet_model = PointNetPMA(input_dim=input_dim, latent_dim=latent_dim)
    state_dict = torch.load(pointnet_model_path)
    # Remove the 'module.' prefix from the state_dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load the modified state_dict into the model
    pointnet_model.load_state_dict(new_state_dict)
    pointnet_model.eval()
    # Load the model weights
    state_dict = torch.load(model_path)
    # Remove the 'module.' prefix from the state_dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    pointnet_model = pointnet_model.to(device)
    # Set the models to evaluation mode
    model.eval()
    pointnet_model.eval()
    # Prepare the dataset and dataloader
    latent_path = 'latent_features.h5'
    if not os.path.exists(latent_path):
        xs_tensor_engineered = advanced_feature_engineering(xs)

        # Initialize PointNetEmbedding model (do this once)
        input_dim = xs_tensor_engineered.shape[-1]
        precompute_latents_to_disk(pointnet_model, 
                                xs_tensor_engineered,
                                latent_path,
                                chunk_size=64)
        del xs_tensor_engineered
        del xs

    dataset = H5Dataset(latent_path)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, 
                            collate_fn=EventDataset.collate_fn, num_workers=0, pin_memory=True, persistent_workers=False)
    
    inference_net = InferenceNet(embedding_dim=latent_dim).to(device)
    # dataset = EventDataset(xs_tensor_engineered.to(device), thetas.to(device), pointnet_model.to(device), device)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return model, pointnet_model, dataloader, device

# Main function to load, evaluate and plot the results
def main():
    model_path = 'final_inference_net.pth'  # Path to the trained model
    pointnet_model_path = 'final_model.pth'  # Path to the pretrained PointNet model
    multi_log=True
    log=False
    # Load the model and data
    model, pointnet_model, dataloader, device = load_model_and_data(model_path, pointnet_model_path)
    true_params = torch.tensor([1.0, 0.5, 2.0, 0.5])
    plot_params_distribution_single(model, pointnet_model, true_params, device, n_mc=1000)
    plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=1000)
    plot_event_histogram(model, pointnet_model, true_params, device, n_mc=100, num_events=1000000)
    plot_loss_curves()
    evaluate_over_n_samples(model, pointnet_model, n=100, num_events=100000, device=device)
if __name__ == "__main__":
    main()