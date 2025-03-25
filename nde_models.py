import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os


# Conditional RealNVP Coupling Layer
class ConditionalRealNVPCouplingLayer(nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.cond_dim = cond_dim

        # Neural network to predict scale and shift (conditioned on x)
        self.scale_shift_net = nn.Sequential(
            nn.Linear(self.half_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.half_dim) * 2)
        )

    def forward(self, theta, x, reverse=False):
        """
        Forward or inverse transformation for the coupling layer.
        """
        theta1, theta2 = theta[:, :self.half_dim], theta[:, self.half_dim:]

        # Concatenate theta1 and x for conditioning
        cond_input = torch.cat([theta1, x], dim=-1)

        # Predict scale and shift
        scale_shift = self.scale_shift_net(cond_input)
        scale = torch.sigmoid(scale_shift[:, :(self.dim - self.half_dim)] + 2.0)
        shift = scale_shift[:, (self.dim - self.half_dim):]

        if not reverse:
            # Forward transformation
            theta2_transformed = theta2 * scale + shift
            theta1_transformed = theta1
            log_det = torch.sum(torch.log(scale), dim=-1)
        else:
            # Inverse transformation
            theta2_transformed = (theta2 - shift) / scale
            theta1_transformed = theta1
            log_det = -torch.sum(torch.log(scale), dim=-1)

        theta_transformed = torch.cat([theta1_transformed, theta2_transformed], dim=-1)
        return theta_transformed, log_det

# Conditional RealNVP Flow
class ConditionalRealNVPFlow(nn.Module):
    def __init__(self, dim, cond_dim, num_layers=4, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        # Stack of coupling layers
        self.layers = nn.ModuleList([
            ConditionalRealNVPCouplingLayer(dim, cond_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, theta, x, reverse=False):
        """
        Forward or inverse transformation for the flow.
        """
        log_det = 0.0
        if not reverse:
            for layer in self.layers:
                theta, ld = layer(theta, x, reverse=False)
                log_det += ld
        else:
            for layer in reversed(self.layers):
                theta, ld = layer(theta, x, reverse=True)
                log_det += ld
        return theta, log_det

class NeuralDensityEstimator(nn.Module):
    def __init__(self, theta_dim, x_dim, num_layers=4, hidden_dim=128):
        super().__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.flow = ConditionalRealNVPFlow(theta_dim, x_dim, num_layers, hidden_dim)

    def forward(self, theta, x):
        return self.log_prob(theta, x)

    def log_prob(self, theta, x):
        # Ensure base distribution is always on the same device as input
        base_dist = torch.distributions.Normal(
            torch.zeros(self.theta_dim, device=theta.device),
            torch.ones(self.theta_dim, device=theta.device)
        )
        z, log_det = self.flow(theta, x, reverse=True)
        log_prob = base_dist.log_prob(z).sum(dim=-1) + log_det
        return log_prob