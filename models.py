import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm as sn
from torch.nn import MultiheadAttention

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        # with torch.no_grad():
        #     self.fc_q.weight.mul_(0.67 * (4 * 6)**-0.25)  # L=6 blocks
        #     self.fc_k.weight.mul_(0.67 * (4 * 6)**-0.25)
        #     self.fc_v.weight.mul_(0.67 * (4 * 6)**-0.25)
        #     self.fc_o.weight.zero_()  # Zero-init final layer

    def forward(self, Q, K):
        # identity = Q
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, hidden_dim=32, num_heads=2, num_inds=512, ln=False):
        super(SetTransformer, self).__init__()

        self.enc = nn.Sequential(
                ISAB(4, hidden_dim, 2, num_inds, ln=ln),
                ISAB(hidden_dim, hidden_dim, 1, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(hidden_dim, 1, 1, ln=ln),
                SAB(hidden_dim, hidden_dim, 1, ln=ln),
                SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
                nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Initial transformation
        x = torch.cat([torch.log(x+1e-8), torch.log10(x+1e-8)], dim=-1).float() #self.input_transform(x)
        # x = torch.log10(x+1e-8).float()
        return self.dec(self.enc(x))

class PointNetCompressor(nn.Module):
    def __init__(self, input_dim=3, latent_dim=4, attn_dim=64, num_heads=1, num_queries=64):
        super(PointNetCompressor, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attn_dim = attn_dim
        self.num_queries = num_queries  # Number of learnable query tokens

        # Shared MLP for feature extraction
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim*2, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, attn_dim, 1),
            nn.BatchNorm1d(attn_dim),
            nn.ReLU()
        )

        # LayerNorm to stabilize self-attention
        self.norm = nn.LayerNorm(attn_dim)

        # Learnable query tokens for efficient self-attention
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, attn_dim))

        # Self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)

        # Bottleneck layer for latent vector
        self.mlp2 = nn.Sequential(
            nn.Linear(attn_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # # Optional T-Net for input transformation
        # self.tnet = nn.Sequential(
        #     nn.Conv1d(input_dim*2, 32, 1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, input_dim * input_dim, 1)
        # )

    def forward(self, x):
        batch_size, num_points, dim = x.size()
        x = torch.cat([torch.log(x+1e-8), torch.log10(x+1e-8)], dim=-1)

        # Reshape for 1D convolution: (batch, dim, num_points)
        x = x.transpose(2, 1)

        # # Input transformation (optional)
        # transform = self.tnet(x)
        # transform = torch.max(transform, dim=2, keepdim=False)[0]
        # transform = transform.view(batch_size, dim, dim)
        # x = torch.bmm(transform, x)  # (batch, dim, num_points)

        # Feature extraction
        x = self.mlp1(x)  # (batch, attn_dim, num_points)

        # Reshape for attention: (batch, num_points, attn_dim)
        x = x.transpose(1, 2)

        # Normalize before self-attention to prevent NaNs
        x = self.norm(x)

        # Downsample: select fixed number of query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)  # (batch, num_queries, attn_dim)
        
        # Self-attention with queries
        attn_output, _ = self.attn(query_tokens, x, x)  # (batch, num_queries, attn_dim)

        # Aggregate attention outputs across query tokens (mean pooling)
        x = attn_output.mean(dim=1)  # (batch, attn_dim)

        # Latent vector
        z = self.mlp2(x)  # (batch, latent_dim)
        return z

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