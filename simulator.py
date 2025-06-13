import torch
from torch.distributions import Uniform, Distribution
import numpy as np

class SimplifiedDIS:
    def __init__(self, device):
        self.Nu = 1
        self.au = 1 # params[0]
        self.bu = 1 # params[1]
        self.Nd = 2
        self.ad = 1 # params[2]
        self.bd = 1 # params[3]
        self.device = None
    def __call__(
        self,
        depth_profiles: np.ndarray,
    ):
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
        self.init(torch.tensor(params))

        xs_p = torch.rand(nevents)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)  # Replace NaNs with 0
        
        xs_n = torch.rand(nevents)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)
        
        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()

class RealisticDISSimulator:
    def __init__(self, device=None, smear=True, smear_std=0.05):
        self.device = device or torch.device("cpu")
        self.smear = smear
        self.smear_std = smear_std
        self.Q0_squared = 1.0  # GeV^2 reference scale
        self.params = None

    def __call__(self, params, nevents=1000):
        return self.sample(params, nevents)

    def init(self, params):
        # Accepts raw list or tensor of 6 params: [logA0, delta, a, b, c, d]
        p = torch.tensor(params, dtype=torch.float32, device=self.device)
        self.logA0 = p[0]
        self.delta = p[1]
        self.a = p[2]
        self.b = p[3]
        self.c = p[4]
        self.d = p[5]

    def q(self, x, Q2):
        A0 = torch.exp(self.logA0)
        scale = (Q2 / self.Q0_squared).clamp(min=1e-6)
        A_Q2 = A0 * scale ** self.delta
        shape = x ** self.a * (1 - x) ** self.b * (1 + self.c * x + self.d * x ** 2)
        return A_Q2 * shape

    def F2(self, x, Q2):
        return x * self.q(x, Q2)

    def sample(self, params, nevents=1000, x_range=(1e-3, 0.9), Q2_range=(1.0, 1000.0)):
        self.init(params)

        # Sample x ~ Uniform, Q2 ~ LogUniform
        x = torch.rand(nevents, device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
        logQ2 = torch.rand(nevents, device=self.device) * (
            np.log10(Q2_range[1]) - np.log10(Q2_range[0])
        ) + np.log10(Q2_range[0])
        Q2 = 10 ** logQ2

        f2 = self.F2(x, Q2)

        if self.smear:
            f2 = f2 + self.smear_std * f2 * torch.randn_like(f2)

        return torch.stack([x, Q2, f2], dim=1)  # shape: [nevents, 3]