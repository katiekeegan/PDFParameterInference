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
from simulator import SimplifiedDIS, up, down, advanced_feature_engineering

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

def evaluate_over_n_parameters(model, pointnet_model, n=100, num_events=100000, device=None):
    """
    Evaluate the model over n true parameter samples and compute errors and chi-squared statistics.
    Args:
        model: The trained model to evaluate.
        pointnet_model: The PointNet model for feature extraction.
        n (int): Number of samples to evaluate.
        num_events (int): Number of events to simulate for each sample.
        device: Device to run the evaluation on (CPU or GPU).
    """     
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
    print(f"Median Chi² up: {np.median(chi2_up):.4f} ± {chi2_up.std():.4f}")
    print(f"Median Chi² down: {np.median(chi2_down):.4f} ± {chi2_down.std():.4f}")

    chi2_up_clip = np.percentile(chi2_up, 99)
    chi2_down_clip = np.percentile(chi2_down, 99)
    
    plt.figure(figsize=(10, 5))
    plt.hist(chi2_up[chi2_up < chi2_up_clip], bins=50, alpha=0.6, label='Chi² Up')
    plt.hist(chi2_down[chi2_down < chi2_down_clip], bins=50, alpha=0.6, label='Chi² Down')
    plt.legend()
    plt.title("Chi² Statistic Distribution (Clipped at 99th percentile)")
    plt.xlabel("Chi²")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("chisq_distributions_clipped.png")
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
    axs[0].scatter(true_events_np[:, 0], true_events_np[:, 1], color='turquoise', alpha=0.2)
    axs[0].set_title(r"$\Xi_{\theta^{*}}$")
    axs[0].set_xlabel(r"$x_{u} \sim u(x|\theta^{*})$")
    axs[0].set_ylabel(r"$x_{d} \sim d(x|\theta^{*})$")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    axs[1].scatter(generated_events_np[:, 0], generated_events_np[:, 1], color='darkorange', alpha=0.2)
    axs[1].set_title(r"$\Xi_{\hat{\theta}}$")
    axs[1].set_xlabel(r"$x_{u} \sim u(x|\hat{\theta})$")
    axs[1].set_ylabel(r"$x_{d} \sim d(x|\hat{\theta})$")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("histograms.png")
    plt.show()

def plot_params_distribution_single(
    model,
    pointnet_model,
    true_params,
    device,
    n_mc=100,
    compare_with_sbi=False,
    sbi_posteriors=None,  # list of tensors
    sbi_labels=None,      # list of strings
    save_path="Dist.png"
):
    model.eval()
    pointnet_model.eval()
    simulator = SimplifiedDIS(torch.device('cpu'))

    true_params = true_params.to(device)

    # Simulate data + feature engineering
    xs = simulator.sample(true_params.cpu(), 100000).to(device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # MC Dropout samples
    samples = sample_with_mc_dropout(model, latent_embedding, n_samples=n_mc).squeeze()

    # Combine all posterior samples for consistent x-limits
    all_samples = [samples.cpu()]
    if compare_with_sbi and sbi_posteriors is not None:
        all_samples.extend([s.cpu() for s in sbi_posteriors])
    
    n_params = true_params.size(0)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    colors = ['skyblue', 'orange', 'green', 'purple', 'gray']
    
    for i in range(n_params):
        # Compute global min/max across all samples for this parameter
        param_vals = [s[:, i].numpy() for s in all_samples]
        xmin = min([v.min() for v in param_vals])
        xmax = max([v.max() for v in param_vals])
        padding = 0.05 * (xmax - xmin)
        xmin -= padding
        xmax += padding

        # MC Dropout
        axes[i].hist(samples[:, i].cpu().numpy(), bins=20, alpha=0.6, density=True, color=colors[0], label='Ours')

        # SBI posteriors
        if compare_with_sbi and sbi_posteriors is not None and sbi_labels is not None:
            for j, sbi_samples in enumerate(sbi_posteriors):
                label = sbi_labels[j] if j < len(sbi_labels) else f"SBI {j}"
                axes[i].hist(
                    sbi_samples[:, i].cpu().numpy(),
                    bins=20, alpha=0.4, density=True,
                    color=colors[(j + 1) % len(colors)],
                    label=label
                )

        # True value
        axes[i].axvline(true_params[i].item(), color='red', linestyle='dashed', label='True Value')
        axes[i].set_xlim(xmin, xmax)
        axes[i].set_title(f'Param {i+1}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=100):
    """
    Plot the PDF distribution of the model's predictions compared to the true parameters.
    Args:
        model: The trained model to evaluate.
        pointnet_model: The PointNet model for feature extraction (stage 2).
        true_params: True parameters for the simulator.
        device: Device to run the evaluation on (CPU or GPU).
        n_mc (int): Number of Monte Carlo samples to draw.
    """
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
    ax_up.plot(x_vals.cpu(), true_up_vals.cpu(), label=r"$u(x|\theta^{*})$", color='blue')
    ax_up.plot(x_vals.cpu(), median_up_vals.cpu(), label=r"Median $u(x|\hat{\theta})$", color='red', linestyle='--')
    ax_up.fill_between(
        x_vals.cpu(),
        lower_up.cpu(),
        upper_up.cpu(),
        color='red',
        alpha=0.3,
        label="10th-90th Percentile"
    )
    ax_up.set_xlabel(r"$x$")
    ax_up.set_xscale("log")
    ax_up.set_ylabel(r"$u(x|\theta)$")
    # ax_up.set_yscale("log")
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
    ax_down.plot(x_vals.cpu(), true_down_vals.cpu(), label=r"$d(x|\theta^{*})$", color='blue')
    ax_down.plot(x_vals.cpu(), median_down_vals.cpu(), label=r"Median $d(x|\hat{\theta})$", color='red', linestyle='--')
    ax_down.fill_between(
        x_vals.cpu(),
        lower_down.cpu(),
        upper_down.cpu(),
        color='red',
        alpha=0.3,
        label="10th-90th Percentile"
    )
    ax_down.set_xlabel("$x$")
    ax_down.set_xscale("log")
    ax_down.set_ylabel(r"$d(x|\theta)$")
    # ax_down.set_yscale("log")
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
    latent_dim = 256  # Example latent dimension
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

    return model, pointnet_model, dataloader, device

# Main function to load, evaluate and plot the results
def main():
    model_path = 'final_inference_net.pth'  # Path to the trained model
    pointnet_model_path = 'final_model.pth'  # Path to the pretrained PointNet model
    # TODO: add RealisticDIS support
    n_mc=100
    # Load the model and data
    model, pointnet_model, dataloader, device = load_model_and_data(model_path, pointnet_model_path)
    true_params = torch.tensor([1.0, 0.5, 1.2, 0.5])
    # plot_params_distribution_single(model, pointnet_model, true_params, device, n_mc=100)
    # Load approximate posteriors generated with SBI methods
    samples_snpe = torch.tensor(np.loadtxt("samples_snpe.txt"), dtype=torch.float32)
    samples_wass = torch.tensor(np.loadtxt("samples_wasserstein.txt"), dtype=torch.float32)
    samples_mmd = torch.tensor(np.loadtxt("samples_mmd.txt"), dtype=torch.float32)
    # Plot the results
    plot_params_distribution_single(
    model=model,
    pointnet_model=pointnet_model,
    true_params=true_params,
    device=device,
    n_mc=n_mc,
    compare_with_sbi=True,
    sbi_posteriors=[samples_snpe, samples_mmd, samples_wass],
    sbi_labels=["SNPE", "MCABC", "Wasserstein MCABC"]
    )
    # Plot the distribution over PDFs and obtain median PDF
    plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=n_mc)
    # Plot the event histograms (cross-sections)
    plot_event_histogram(model, pointnet_model, true_params, device, n_mc=n_mc, num_events=1000000)
    # Plot the loss curves
    plot_loss_curves()
    # Evaluate the model over n true parameters and compute errors and chi-squared statistics
    evaluate_over_n_parameters(model, pointnet_model, n=n_mc, num_events=100000, device=device)
if __name__ == "__main__":
    main()