import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset

class SimplifiedDIS:
    def __init__(self, device=None):
        self.Nu = 1
        self.au = 1  # params[0]
        self.bu = 1  # params[1]
        self.Nd = 2
        self.ad = 1  # params[2]
        self.bd = 1  # params[3]
        self.device = device

    def __call__(self, depth_profiles: np.ndarray):
        """Call to simulator."""
        return self.sample(depth_profiles)

    def init(self, params):
        self.Nu = 1
        self.au = params[0]
        self.bu = params[1]
        self.Nd = 2
        self.ad = params[2]
        self.bd = params[3]

    def up(self, x):
        u = self.Nu * (x ** self.au) * ((1 - x) ** self.bu)
        return u

    def down(self, x):
        d = self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)
        return d

    def sample(self, params, nevents=1):
        self.init(torch.tensor(params, device=self.device))

        xs_p = torch.rand(nevents, device=self.device)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)  # Replace NaNs with 0

        xs_n = torch.rand(nevents, device=self.device)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()


def generate_data(num_samples, num_events, theta_dim=4, x_dim=2, device=torch.device("cpu")):
    """
    Generate a dataset of (theta, x) pairs using the simulator.
    theta: (num_samples, theta_dim)
    x: (num_samples, num_events, x_dim)
    """
    simulator = SimplifiedDIS(device)
    
    # Define the parameter ranges for the thetas
    ranges = [(-10, 10), (-10,10), (-10,10), (-10,10)]  # Example ranges
    
    # Generate thetas within the defined ranges
    thetas = np.column_stack([np.random.uniform(low, high, size=num_samples) for low, high in ranges])
    
    # Generate xs based on the thetas using the simulator
    xs = np.array([simulator.sample(theta, num_events).cpu().numpy() for theta in thetas]) + 1e-8
    
    # Convert the generated numpy arrays to PyTorch tensors
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32, device=device)
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    
    return thetas_tensor, xs_tensor
class PointNetEmbedding(nn.Module):
    def __init__(self, input_dim=2, latent_dim=64, outlier_attention_factor=10.0):
        super(PointNetEmbedding, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.outlier_attention_factor = outlier_attention_factor
        
        # MLP layers for point feature transformation
        self.mlp1 = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Final layers to get latent representation
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim)
        )

    def forward(self, x):
        """
        Forward pass for embedding the input events into a latent vector.
        x: Tensor of shape (batch_size, num_events, input_dim)
        """
        # Ensure the input shape is as expected
        batch_size, num_events, _ = x.shape  # Get the batch size and number of events

        # Transform each point using the MLP
        x = self.mlp1(x)  # Shape will be (batch_size, num_events, 256)
        # Apply Max pooling for permutation invariance
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 256, num_events)
        # Take the max across the num_events dimension (dimension 1)
        x, _ = torch.max(x, dim=-1)  # Shape becomes (batch_size, 256)
        x = x.squeeze(-1)  # Shape becomes (batch_size, 256)
        # Apply final MLP to get latent vector
        latent = self.mlp2(x)  # Shape becomes (batch_size, latent_dim)
        
        # Attention for outliers (use a simple attention mechanism for now)
        attention = torch.sigmoid(latent) * self.outlier_attention_factor
        latent = latent * attention  # Apply attention to give more weight to outliers
        
        return latent  # Shape will be (batch_size, latent_dim)


class LatentToParamsNN(nn.Module):
    def __init__(self, latent_dim, param_dim):
        super(LatentToParamsNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(latent_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)          # Second fully connected layer
        self.fc3 = nn.Linear(64, param_dim)    # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation after second layer
        x = 10*torch.tanh(self.fc3(x))              # Output layer (no activation, linear output)
        return x

class EventDataset(Dataset):
    def __init__(self, event_data, param_data, pointnet_model, device):
        self.event_data = event_data
        self.param_data = param_data
        self.pointnet_model = pointnet_model
        self.device = device

    def __len__(self):
        return len(self.event_data)

    def __getitem__(self, idx):
        events = self.event_data[idx].to(self.device)  # Move the events tensor to the correct device
        params = self.param_data[idx].to(self.device)  # Move the params tensor to the correct device
        
        # Get latent embedding from pretrained PointNet model
        latent_embedding = self.pointnet_model(events.unsqueeze(0).to(self.device))  # Ensure it is on the correct device
        return latent_embedding, params


# Training Function
def train(model, pointnet_model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for latent_embeddings, true_params in dataloader:
        latent_embeddings = latent_embeddings.to(device)
        true_params = true_params.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass through LatentToParamsNN
        predicted_params = model(latent_embeddings)

        # Compute loss
        loss = criterion(predicted_params.squeeze(), true_params.squeeze())
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    # Assuming latent_dim and param_dim are defined
    latent_dim = 64  # Example latent dimension

    # Instantiate the model
    model = LatentToParamsNN(latent_dim, 4)
    # Example usage:
    # Simulate some data for `xs`
    num_samples = 300
    num_events = 10000
    thetas, xs = generate_data(num_samples, num_events)
    multi_log=True
    log=False
    # Convert `xs` to torch tensor
    xs_tensor = torch.tensor(xs, dtype=torch.float32)
    if multi_log:
        xs_tensor = torch.cat([torch.log(xs_tensor + 1e-8), torch.log10(xs_tensor + 1e-8)], dim=-1).float()
    elif log:
        xs_tensor = torch.log(xs_tensor + 1e-8)
    # Example input: latent embeddings of shape (batch_size, latent_dim)
    # Create the model again
    pointnet_model = PointNetEmbedding(input_dim=xs_tensor.shape[-1], latent_dim=64)

    # Load the saved state_dict into the model
    pointnet_model.load_state_dict(torch.load('pointnet_embedding.pth'))
    pointnet_model.eval()  

    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pointnet_model.to(device)

    # Create Dataset and DataLoader
    dataset = EventDataset(xs_tensor, thetas, pointnet_model, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_epochs = 10000  # Set the number of epochs
    for epoch in range(num_epochs):
        avg_loss = train(model, pointnet_model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'latent_to_params_model.pth')
    evaluate_and_plot(model, pointnet_model, dataloader, device)

import matplotlib.pyplot as plt

def evaluate_and_plot(model, pointnet_model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    predicted_params_list = []
    true_params_list = []

    # Iterate over the dataset
    with torch.no_grad():  # No need to compute gradients for evaluation
        for latent_embeddings, true_params in dataloader:
            latent_embeddings = latent_embeddings.to(device)
            true_params = true_params.to(device)

            # Forward pass through LatentToParamsNN
            predicted_params = model(latent_embeddings)

            # Store the true and predicted parameters
            predicted_params_list.append(predicted_params)
            true_params_list.append(true_params)

    # Convert lists to tensors
    predicted_params = torch.cat(predicted_params_list, dim=0)
    true_params = torch.cat(true_params_list, dim=0)

    # Compute the errors
    errors = predicted_params - true_params  # Shape: (num_samples, param_dim)

    # Plotting the error distribution for each parameter
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 4 subplots for the 4 parameters
    for i in range(4):
        ax = axes[i]
        ax.hist(errors[:, i].cpu().numpy(), bins=50, alpha=0.7)
        ax.set_title(f'Error Distribution for Parameter {i+1}')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    plt.savefig('Error_Distribution.png')
if __name__ == "__main__":
    main()
    # After training, call the evaluate_and_plot function