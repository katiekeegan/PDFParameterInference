import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from params_learning import LatentToParamsNN, PointNetEmbedding, EventDataset, generate_data  # Adjust the import as needed

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
        ax.hist(errors[:, i].cpu().numpy(), bins=15, alpha=0.7)
        ax.set_title(f'Error Distribution for Parameter {i+1} (|theta - theta*|/|theta*|)')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    plt.savefig('Error_Distribution.png')


# Load the model and data
def load_model_and_data(model_path, pointnet_model_path, num_samples=100, num_events=10000, device=None):
    # Ensure the device is set
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the models
    latent_dim = 64  # Example latent dimension
    model = LatentToParamsNN(latent_dim, 4)  # Parameter dimension is 4
    pointnet_model = PointNetEmbedding(input_dim=4, latent_dim=latent_dim)

    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    pointnet_model.load_state_dict(torch.load(pointnet_model_path))

    # Set the models to evaluation mode
    model.eval()
    pointnet_model.eval()

    # Generate test data
    thetas, xs = generate_data(num_samples, num_events, device=device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    multi_log=True
    log=False

    # Convert `xs` to torch tensor
    xs_tensor = torch.tensor(xs, dtype=torch.float32)
    if multi_log:
        xs_tensor = torch.cat([torch.log(xs_tensor + 1e-8), torch.log10(xs_tensor + 1e-8)], dim=-1).float()
    elif log:
        xs_tensor = torch.log(xs_tensor + 1e-8)
    # Initialize the model
    
    # Prepare the dataset and dataloader
    dataset = EventDataset(xs_tensor, thetas, pointnet_model, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return model, pointnet_model, dataloader, device

# Main function to load, evaluate and plot the results
def main():
    model_path = 'latent_to_params_model.pth'  # Path to the trained model
    pointnet_model_path = 'pointnet_embedding.pth'  # Path to the pretrained PointNet model

    # Load the model and data
    model, pointnet_model, dataloader, device = load_model_and_data(model_path, pointnet_model_path)

    # Evaluate the model and plot the error distribution
    evaluate_and_plot(model, pointnet_model, dataloader, device)

if __name__ == "__main__":
    main()
