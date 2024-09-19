
import torch
import torch.nn as nn
from torch import cat
import torch.nn.functional as F

class ConvolutionalGAN(nn.Module):
    def __init__(self, noise_dim, param_dims):
        super().__init__()
        # Encoder
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 16),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(),
            nn.Linear(16, int(param_dims)),
            nn.ReLU()
        )
    # Function to initialize weights with He initialization
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x1 = x
        x2 = self.model(x1)
        # x2 = x2**2
        # x2[...,0] = 5.5*nn.Tanh()(x2[...,0])+4.5
        # x2[...,1] = 5*nn.Tanh()(x2[...,1])+5
        # x2[...,2] = 5*nn.Tanh()(x2[...,2])+5
        # x2[...,3] = 30*nn.Tanh()(x2[...,3])
        return x2

class SurrogatePhysicsModel(nn.Module):
    def __init__(self, input_dim):
        super(SurrogatePhysicsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # First fully connected layer
        self.fc2 = nn.Linear(32, 16)  # Second fully connected layer
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    # Function to initialize weights with He initialization
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # Apply Sigmoid activation function
        return x


class MLP(nn.Module):
    def __init__(self, input_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)  # First fully connected layer
        self.fc2 = nn.Linear(16, 8)  # Second fully connected layer
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
            # Function to initialize weights with He initialization
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)
        x = self.sigmoid(x)  # Apply Sigmoid activation function
        return x

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Transpose for multihead attention (batch_first=False for nn.MultiheadAttention)
        x = x.transpose(0, 1)  # Shape: (set_size, batch_size, hidden_dim)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        return x.transpose(0, 1)  # Shape: (batch_size, set_size, hidden_dim)
class DeepSetsWithAttention(nn.Module):
    def __init__(self, batch_size=5,input_dim=1, hidden_dim=8, output_dim=1, num_heads=4):
        super(DeepSetsWithAttention, self).__init__()
        self.input_dim = input_dim
        self.element_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BLazyatchNorm1d(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.self_attention = SelfAttention(hidden_dim, num_heads)
        self.aggregation_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        input_dim = self.input_dim
        # input_dim = 
        batch_size, set_size, _ = x.size()

        x = x.view(-1, input_dim)  # Shape: (batch_size * set_size, input_dim)
        x = self.element_nn(x)  # Shape: (batch_size * set_size, hidden_dim)
        x = x.view(batch_size, set_size, -1)  # Shape: (batch_size, set_size, hidden_dim)

        # x = self.self_attention(x)  # Shape: (batch_size, set_size, hidden_dim)

        x = x.sum(dim=1)  # Aggregation using sum (batch_size, hidden_dim)

        # x = self.dropout(x)
        x = self.aggregation_nn(x)  # Shape: (batch_size, output_dim)

        return x


class PointDiscriminator(nn.Module):
    def __init__(self, loglog=False):
        super(PointDiscriminator, self).__init__()
        # Define your discriminator network architecture here
        self.fc1 = nn.Linear(in_features=1, out_features=8)  # Assuming 3D points
        self.fc2 = nn.Linear(in_features=8, out_features=1)  # Output a single scalar for each point

        self.sigmoid = nn.Sigmoid()
        self.loglog = False
    def forward(self, point_cloud):
        if self.loglog:
            point_cloud = torch.log(point_cloud+1e-10)
        # point_cloud shape: (batch_size, num_points, 3)
        batch_size, num_points, _  = point_cloud.size()

        # Flatten the point cloud to process each point individually
        x = point_cloud.view(batch_size * num_points, -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)

        # Reshape back to (batch_size, num_points, 1)
        x = x.view(batch_size, num_points, 1)

        # Aggregate classification across points (mean or sum)
        # Here, using mean to get a probability for the entire point cloud
        point_cloud_decision = torch.mean(x, dim=1)  # shape: (batch_size, 1)

        return point_cloud_decision