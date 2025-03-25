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
