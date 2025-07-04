import torch
from torch.distributions import Uniform, Distribution
import numpy as np

class Gaussian2DSimulator:
    """
    Unimodal 2D Gaussian simulator with 1D parameter vector input.
    Parameter vector: [mu_x, mu_y, sigma_x, sigma_y, rho]
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    def sample(self, theta, nevents=1000):
        """
        theta: torch.tensor of shape (5,) -- [mu_x, mu_y, sigma_x, sigma_y, rho]
        Returns: torch.tensor of shape (nevents, 2)
        """
        mu_x, mu_y, sigma_x, sigma_y, rho = theta
        mean = torch.tensor([mu_x, mu_y], device=self.device)
        cov = torch.tensor([
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2]
        ], device=self.device)
        samples = torch.distributions.MultivariateNormal(mean, cov).sample((nevents,))
        return samples

class SimplifiedDIS:
    def __init__(self, device=None):
        self.device = device

    def init(self, params):
        self.Nu = 1
        self.au = params[0]
        self.bu = params[1]
        self.Nd = 2
        self.ad = params[2]
        self.bd = params[3]

    def up(self, x):
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    def sample(self, params, nevents=1):
        self.init(torch.tensor(params, device=self.device))

        xs_p = torch.rand(nevents, device=self.device)
        sigma_p = 4 * self.up(xs_p) + self.down(xs_p)
        sigma_p = torch.nan_to_num(sigma_p, nan=0.0)

        xs_n = torch.rand(nevents, device=self.device)
        sigma_n = 4 * self.down(xs_n) + self.up(xs_n)
        sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

        return torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()

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
    
class RealisticDIS:
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
        scale_factor = (Q2 / self.Q0_squared).clamp(min=1e-6)
        A_Q2 = A0 * scale_factor ** self.delta
        shape = x.clamp(min=1e-6, max=1.0)**self.a * (1 - x.clamp(min=0.0, max=1.0))**self.b
        poly = 1 + self.c * x + self.d * x**2
        shape = shape * poly.clamp(min=1e-6)  # avoid negative polynomial tail
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
            noise = torch.randn_like(f2) * (self.smear_std * f2)
            f2 = f2 + noise
            f2f = f2.clamp(min=1e-6)

        return torch.stack([x, Q2, f2], dim=1)  # shape: [nevents, 3]