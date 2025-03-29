import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from params_learning import LatentToParamsNN, PointNetEmbedding, EventDataset, generate_data, MDN  # Adjust the import as needed
from simulator import *
from models import *

def up(x, params):
    return (x ** params[0]) * ((1 - x) ** params[1])
    
def down(x, params):
    return (x ** params[2]) * ((1 - x) ** params[3])



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


def load_and_plot(loss_history_path):
    # Load loss history
    loss_history = np.load(loss_history_path)
    
    # Plot training curve
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
    Plots loss history with optional smoothing
    
    Args:
        loss_history: numpy array of loss values
        smooth_window: size of moving average window (set to 1 for no smoothing)
    """
    plt.figure(figsize=(12, 7))
    
    # Raw loss
    plt.plot(loss_history, 'b', alpha=0.3, label='Raw loss')
    
    # Smoothed loss
    if smooth_window > 1:
        smooth_loss = np.convolve(loss_history, np.ones(smooth_window)/smooth_window, mode='valid')
        plt.plot(range(smooth_window-1, len(loss_history)), smooth_loss, 
                'r', linewidth=2, label=f'Smoothed (window={smooth_window})')
    
    plt.title('Training Loss History', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some analysis markers
    min_loss = np.min(loss_history)
    min_epoch = np.argmin(loss_history)
    plt.scatter(min_epoch, min_loss, c='green', s=100, label=f'Min loss: {min_loss:.4f}')
    
    plt.legend()
    plt.show()
    
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Minimum loss: {min_loss:.4f} at epoch {min_epoch}")

def evaluate_and_plot(model, pointnet_model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    predicted_params_list = []
    true_params_list = []

    # Ensure the model is on the correct device
    model.to(device)
    pointnet_model.to(device)

    # Iterate over the dataset
    with torch.no_grad():  # No need to compute gradients for evaluation
        for latent_embeddings, true_params in dataloader:
            # Move data to the correct device
            latent_embeddings = latent_embeddings.to(device)
            true_params = true_params.to(device)

            # Forward pass through LatentToParamsNN
            predicted_params = model(latent_embeddings)

            # Store the true and predicted parameters
            predicted_params_list.append(predicted_params)
            true_params_list.append(true_params)

    # Convert lists to tensors
    predicted_params = torch.cat(predicted_params_list, dim=0).squeeze()
    true_params = torch.cat(true_params_list, dim=0)

    # Compute the errors
    errors = torch.abs(predicted_params - true_params)/torch.abs(true_params)  # Shape: (num_samples, param_dim)

    # Plotting the error distribution for each parameter
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 4 subplots for the 4 parameters
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
    For each instance and each parameter, sample from the mixture.
    pi, mu, sigma: tensors with shape (batch, param_dim, num_components)
    Returns: sampled parameters with shape (batch, param_dim)
    """
    batch_size, param_dim, num_components = mu.shape
    sampled_params = []
    for b in range(batch_size):
        params_instance = []
        for p in range(param_dim):
            # Get the mixture weights, means, and sigmas for this parameter
            pi_p = pi[b, p, :]  # Shape: (num_components,)
            mu_p = mu[b, p, :]
            sigma_p = sigma[b, p, :]
            # Sample an index from the mixture components
            component = torch.multinomial(pi_p, 1)
            # Draw a sample from the chosen Gaussian
            sample_val = torch.normal(mu_p[component], sigma_p[component])
            params_instance.append(sample_val.squeeze())
        sampled_params.append(torch.stack(params_instance))
    return torch.stack(sampled_params)  # (batch, param_dim)

def evaluate_and_plot_samples(model, pointnet_model, dataloader, device, n_mc=10):
    model.eval()
    predicted_samples_list = []
    true_params_list = []

    with torch.no_grad():
        for latent_embeddings, true_params in dataloader:
            latent_embeddings = latent_embeddings.to(device)
            true_params = true_params.to(device)
            
            # Get MDN outputs
            pi, mu, sigma = model(latent_embeddings)
            
            # For each latent embedding, draw multiple samples
            batch_samples = []
            for _ in range(n_mc):
                sampled = sample_from_mdn(pi, mu, sigma)  # (batch, param_dim)
                batch_samples.append(sampled.unsqueeze(0))
            # Average the samples along the MC dimension or keep them all to plot distributions
            # Here we concatenate to see the full spread: shape becomes (n_mc, batch, param_dim)
            batch_samples = torch.cat(batch_samples, dim=0)
            predicted_samples_list.append(batch_samples)  # List of (n_mc, batch, param_dim)
            true_params_list.append(true_params.unsqueeze(0).expand(n_mc, -1, -1))
    
    # Concatenate over batches and Monte Carlo samples
    predicted_samples = torch.cat(predicted_samples_list, dim=1)  # (n_mc, total_samples, param_dim)
    true_params = torch.cat(true_params_list, dim=1)  # (n_mc, total_samples, param_dim)
    
    # Now, for each parameter, plot a histogram of the predicted samples
    total_samples = predicted_samples.shape[1]
    fig, axes = plt.subplots(1, true_params.shape[2], figsize=(20, 5))
    for i in range(true_params.shape[2]):
        ax = axes[i]
        # Flatten the samples for parameter i
        samples = predicted_samples[:, :, i].reshape(-1).cpu().numpy()
        ax.hist(samples, bins=50, alpha=0.7)
        # Draw the true value as a vertical line
        # Here, we assume the true parameter is constant across samples (e.g., provided externally)
        # You may adjust this if true_params vary per instance.
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
    """
    Plots the predicted parameters distribution for a single set of true parameters.
    
    Args:
        model: Your MDN-based model.
        pointnet_model: The model to get the latent embedding (if needed).
        true_params: A tensor of shape (1, param_dim) representing the fixed true parameters.
        device: Device to run the computations on.
        n_mc: Number of Monte Carlo samples to draw.
    """
    model.eval()
    pointnet_model.eval()
    
    # For a single set of true parameters, you need a corresponding latent embedding.
    # This could come from an input that you know corresponds to true_params.
    # For illustration, we assume you have a representative input 'xs_tensor'
    # that generates a latent embedding via your PointNet. If you don't have that,
    # you could simply use a random latent vector as a proxy.
    #
    # Here, we'll assume a random latent embedding as a placeholder:
    true_params = true_params.to(device)
    simulator = SimplifiedDIS(device)
    num_events = 100000
    xs = simulator.sample(true_params, num_events)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # latent_embedding = pointnet_model(xstorch.randn(1, 128).to(device)
    
    # Perform Monte Carlo sampling from the MDN for this latent embedding.
    samples = []
    with torch.no_grad():
        for _ in range(n_mc):
            pi, mu, sigma = model(latent_embedding)
            sampled = sample_from_mdn(pi, mu, sigma)  # shape: (1, param_dim)
            samples.append(sampled.squeeze(0))
    
    samples = torch.stack(samples)  # shape: (n_mc, param_dim)
    # Plot histograms for each parameter.
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
    """
    Plots the predicted PDF distribution for a single set of true parameters.
    
    Args:
        model: Your MDN-based model.
        pointnet_model: The model to get the latent embedding (if needed).
        true_params: A tensor of shape (1, param_dim) representing the fixed true parameters.
        device: Device to run the computations on.
        n_mc: Number of Monte Carlo samples to draw.
    """
    model.eval()
    pointnet_model.eval()
    
    # For a single set of true parameters, you need a corresponding latent embedding.
    # This could come from an input that you know corresponds to true_params.
    # For illustration, we assume you have a representative input 'xs_tensor'
    # that generates a latent embedding via your PointNet. If you don't have that,
    # you could simply use a random latent vector as a proxy.
    #
    # Here, we'll assume a random latent embedding as a placeholder:
    true_params = true_params.to(device)
    simulator = SimplifiedDIS(device)
    num_events = 100000
    xs = simulator.sample(true_params, num_events)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    latent_embedding = pointnet_model(xs_tensor.unsqueeze(0))

    # latent_embedding = pointnet_model(xstorch.randn(1, 128).to(device)
    
    # Perform Monte Carlo sampling from the MDN for this latent embedding.
    samples = []
    with torch.no_grad():
        for _ in range(n_mc):
            pi, mu, sigma = model(latent_embedding)
            sampled = sample_from_mdn(pi, mu, sigma)  # shape: (1, param_dim)
            samples.append(sampled.squeeze(0))
    
    samples = torch.stack(samples)  # shape: (n_mc, param_dim)
    # Plot histograms for each parameter.
    x_grid = torch.linspace(0, 1, 100).to(device)  # 100 points between 0 and 1

    # Evaluate the function for all samples and x values
    num_samples = samples.shape[0]
    function_values = torch.zeros((num_samples, len(x_grid))).to(device)

    for i in range(num_samples):
        params = samples[i]  # Get the i-th parameter sample
        function_values[i] = down(x_grid, params)  # Evaluate up(x, params) for all x in x_grid

    # Compute statistics
    mean_f = torch.mean(function_values, dim=0)  # Mean across samples
    std_f = torch.std(function_values, dim=0)    # Std deviation across samples

    # Evaluate the true function
    true_function = down(x_grid, true_params)

    # Optional: Compute quantiles (e.g., 5% and 95%)
    lower = torch.quantile(function_values, 0.05, dim=0)
    upper = torch.quantile(function_values, 0.95, dim=0)

    # Convert to numpy for plotting
    x_grid_np = x_grid.cpu().numpy()
    mean_f_np = mean_f.cpu().numpy()
    std_f_np = std_f.cpu().numpy()
    lower_np = lower.cpu().numpy()
    upper_np = upper.cpu().numpy()
    true_function_np = true_function.cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid_np, true_function_np, label="True function", color="red", linestyle="--", linewidth=2)
    plt.plot(x_grid_np, mean_f_np, label="Mean function", color="blue")
    plt.fill_between(
        x_grid_np,
        mean_f_np - std_f_np,
        mean_f_np + std_f_np,
        alpha=0.3,
        color="blue",
        label="±1 std dev",
    )
    # Alternatively, plot quantiles:
    # plt.fill_between(x_grid_np, lower_np, upper_np, alpha=0.3, color="blue", label="90% CI")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Distribution of Functions Induced by Parameter Samples")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('down.png')


def plot_params_distribution(model, pointnet_model, true_params, dataloader, device):
    model.train()  # Keep dropout active

    num_events = 100000

    # Ensure the model is on the correct device
    model.to(device)
    pointnet_model.to(device)
    true_params = true_params.to(device)
    simulator = SimplifiedDIS(device)
    xs = simulator.sample(true_params, num_events)

    multi_log=True
    log=False

    # Convert `xs` to torch tensor
    xs_tensor = torch.tensor(xs, dtype=torch.float32)
    num_samples = 50
    # Generate test data
    thetas, xs = generate_data(num_samples, num_events, device=device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor = advanced_feature_engineering(xs_tensor).to(device)
    
    # Iterate over the dataset
    with torch.no_grad():  # No need to compute gradients for evaluation
        latent_embedding = pointnet_model(xs_tensor)
        
        # Forward pass through LatentToParamsNN
        predicted_params = model(latent_embedding)
        latent_embedding = latent_embedding.to(device)
        preds = torch.stack([model(latent_embedding[i, :].unsqueeze(0)) for i in range(num_samples)], dim=0).squeeze()
    # Convert true parameters to CPU numpy array
    true_params_np = true_params.cpu().numpy()
    # Plotting the error distribution for each parameter
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 4 subplots for the 4 parameters
    for i in range(4):
        ax = axes[i]
        ax.hist(preds[:, i].squeeze().cpu().numpy(), bins=50, alpha=0.7)
        ax.axvline(true_params_np[i], color='r', linestyle='dashed', linewidth=2, label='True Value')
        ax.set_title(f'Parameter {i+1}')
        ax.set_xlabel('Error (|theta - theta*|/|theta*|)')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig('Params_Distribution.png')
    plt.show()


# Load the model and data
def load_model_and_data(model_path, pointnet_model_path, num_samples=100, num_events=10000, device=None):
    # Ensure the device is set
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate test data
    thetas, xs = generate_data(num_samples, num_events, device=device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    xs_tensor_engineered = advanced_feature_engineering(xs)
    input_dim = xs_tensor_engineered.shape[-1]

    # Instantiate the models
    latent_dim = 128  # Example latent dimension
    model = MDN(latent_dim, 4)  # Parameter dimension is 4
    pointnet_model = PointNetEmbedding(input_dim=input_dim, latent_dim=latent_dim)

    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    pointnet_model.load_state_dict(torch.load(pointnet_model_path))
    model = model.to(device)
    pointnet_model = pointnet_model.to(device)
    # Set the models to evaluation mode
    model.eval()
    pointnet_model.eval()
    # Prepare the dataset and dataloader
    dataset = EventDataset(xs_tensor_engineered.to(device), thetas.to(device), pointnet_model.to(device), device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return model, pointnet_model, dataloader, device

# Main function to load, evaluate and plot the results
def main():
    model_path = 'latent_to_params_model.pth'  # Path to the trained model
    pointnet_model_path = 'pointnet_embedding_1000.pth'  # Path to the pretrained PointNet model
    multi_log=True
    log=False
    # Load the model and data
    model, pointnet_model, dataloader, device = load_model_and_data(model_path, pointnet_model_path)
    # true_params = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # Plot loss values
    # Load and plot
    # loss_history = load_and_plot('loss_history.npy')
    # Evaluate the model and plot the error distribution
    # evaluate_and_plot(model, pointnet_model, dataloader, device)
    # Plot distribution of estimated parameters given a specific set of true parameters
    true_params = torch.tensor([1.0, 4.0, 1.0, 4.0])
    # plot_params_distribution(model, pointnet_model, true_params, dataloader, device)
    plot_params_distribution_single(model, pointnet_model, true_params, device, n_mc=1000)
    plot_PDF_distribution_single(model, pointnet_model, true_params, device, n_mc=1000)
if __name__ == "__main__":
    main()
