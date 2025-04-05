import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm as sn
from torch.nn import MultiheadAttention

class HierarchicalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, chunk_size=4096):
        super().__init__()
        self.chunk_size = chunk_size
        # Local attention within chunks
        self.local_attn = nn.Linear(hidden_dim, 1)
        # Global attention over chunk summaries
        self.global_attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, N, D = x.shape
        # Step 1: Split into manageable chunks
        x = x.view(B, -1, self.chunk_size, D)  # [B, num_chunks, chunk_size, D]
        
        # Step 2: Local attention within each chunk
        local_weights = torch.softmax(self.local_attn(x), dim=2)  # [B, num_chunks, chunk_size, 1]
        chunk_summaries = (x * local_weights).sum(dim=2)  # [B, num_chunks, D]
        
        # Step 3: Global attention across chunks
        global_weights = torch.softmax(self.global_attn(chunk_summaries), dim=1)  # [B, num_chunks, 1]
        pooled = (chunk_summaries * global_weights).sum(dim=1)  # [B, D]
        
        return pooled

class PointNetEmbedding(nn.Module):
    def __init__(self, input_dim=2, latent_dim=64, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Enhanced MLP with residual connections
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.GroupNorm(8, hidden_dim),  # Better than BatchNorm for variable-length sequences
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.GroupNorm(8, hidden_dim),
            nn.Sigmoid()
        )
        
        # # Learnable pooling (instead of just max-pooling)
        # self.pool = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 1),
        #     nn.Sigmoid()  # Soft weights for pooling
        # )


        # Hierarchical attention pooling for large point clouds
        self.pool = HierarchicalAttentionPooling(hidden_dim, chunk_size=5000)
        
        # Final MLP with residual
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        batch_size, num_events, _ = x.shape
        
        # Feature extraction
        x = self.mlp1(x)  # (batch_size, num_events, hidden_dim)
        
        # # Learnable pooling (weighted sum instead of max)
        # weights = self.pool(x)  # (batch_size, num_events, 1)
        # x = torch.mean(x * weights, dim=1)  # (batch_size, hidden_dim)
        x = self.pool(x)
        # Final MLP
        latent = self.mlp2(x)  # (batch_size, latent_dim)
        return latent

class DiffusionModel(nn.Module):
    def __init__(self, sample_dim, param_dim, hidden_dim=128, time_embedding_dim=32, sample_encode_dim=32, timesteps=1000, n_events=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        
        self.sample_encoder = PointNetCompressor(input_dim=sample_dim, latent_dim=sample_encode_dim)
        
        self.time_embedding = nn.Linear(1, time_embedding_dim)
        
        self.diffusion_process = nn.Sequential(
            nn.Linear(sample_encode_dim + param_dim + time_embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, param_dim)
        )
    
    def forward(self, samples, t, params):
        t_embedding = self.time_embedding(t.squeeze().unsqueeze(-1).float())  # Simple sinusoidal encoding
        samples_emb = self.sample_encoder(samples).squeeze()
        combined_input = torch.cat([samples_emb, t_embedding, params], dim=-1)
        pred_noise = self.diffusion_process(combined_input)
        return pred_noise

class LatentToParamsNN(nn.Module):
    def __init__(self, latent_dim, param_dim, dropout_prob=0.2):
        super(LatentToParamsNN, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Separate layers for mean and variance
        self.fc_mean = nn.Linear(64, param_dim)
        self.fc_log_var = nn.Linear(64, param_dim)  # Output log variance

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        mean = 10 * torch.sigmoid(self.fc_mean(x))  # Scale to parameter range
        # log_var = self.fc_log_var(x)  # Unconstrained log variance
        log_var = torch.tanh(self.fc_log_var(x), min=-10, max=10)  # Prevent extreme values
        variance = torch.exp(log_var)
        # log_var = torch.tanh(self.fc_logvar(x)) * 5  # Keep within a reasonable range
        return mean, variance