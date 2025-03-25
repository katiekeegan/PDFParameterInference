import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def generate_labels(thetas, max_distance=20.0):
    """
    Generate continuous similarity labels for each pair based on the scaled Euclidean distance 
    between theta values. Return a continuous similarity score in the range [0, 1].
    """
    labels = []
    for i in range(len(thetas)):
        for j in range(i + 1, len(thetas)):
            similarity = torch.norm(thetas[i] - thetas[j]) # Compute distance between thetas
            # similarity = 1.0 - dist/max_distance  # Scale the distance
            labels.append(similarity)
    labels = torch.tensor(labels, dtype=torch.float32)
    labels = labels/labels.max()
    return labels

def contrastive_loss(latent1, latent2, similarity):
    """
    Contrastive loss function with continuous similarity labels, no margin.
    """
    euclidean_distance = torch.norm(latent1 - latent2, p=2)
    loss = 0.5 * (similarity * torch.pow(euclidean_distance, 2) + 
                  (1 - similarity) * torch.pow(euclidean_distance, 2))
    return loss.mean()


# Example training loop
def train(model, thetas, xs, num_epochs=10, batch_size=32, learning_rate=1e-3, margin=1.0):
    """
    Train the PointNet model with contrastive loss.
    """
    # Prepare dataset: Convert thetas and xs into PyTorch tensors
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32)
    xs_tensor = torch.tensor(xs, dtype=torch.float32)
    
    # Create DataLoader
    dataset = TensorDataset(thetas_tensor, xs_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, (theta, x) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get latent vectors for each sample in the batch
            latent_vectors = model(x)  # Latent vectors for batch
            
            # Generate labels based on thetas' similarity
            labels = generate_labels(thetas)  # Adjust threshold as needed
                        
            # Create pairs of latent vectors and corresponding labels
            batch_size = latent_vectors.size(0)
            loss = 0
            pair_idx = 0
            
            for j in range(batch_size):
                for k in range(j + 1, batch_size):  # Generate pairs
                    latent1 = latent_vectors[j]
                    latent2 = latent_vectors[k]
                    label = labels[pair_idx]
                    pair_idx += 1

                    # Compute contrastive loss for the pair
                    loss += contrastive_loss(latent1, latent2, label)
            
            # Backpropagate and update the model
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')


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
# Initialize the model
model = PointNetEmbedding(input_dim=xs_tensor.shape[-1], latent_dim=64)

# Train the model
train(model, thetas, xs_tensor, num_epochs=100, batch_size=32, learning_rate=1e-3, margin=1.0)
# Saving the model
torch.save(model.state_dict(), 'pointnet_embedding.pth')