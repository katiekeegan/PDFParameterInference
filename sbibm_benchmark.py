import torch
from torch.distributions import Uniform
from sbi.inference import simulate_for_sbi, prepare_for_sbi, SNPE, MCABC
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils.torchutils import BoxUniform
import numpy as np
from scipy.stats import wasserstein_distance

import torch
from torch.distributions import Uniform
from sbi.inference import simulate_for_sbi, prepare_for_sbi, MCABC
import numpy as np

########################
# PRIOR OVER PARAMETERS
########################
prior = torch.distributions.Independent(
    Uniform(low=torch.zeros(4), high=5 * torch.ones(4)), 1
)

########################
# KERNEL DISTANCE BETWEEN POINT CLOUDS
########################
def pairwise_squared_dist(x, y):
    # x: [N, D], y: [M, D]
    x2 = x.pow(2).sum(dim=1, keepdim=True)  # [N, 1]
    y2 = y.pow(2).sum(dim=1, keepdim=True).T  # [1, M]
    xy = x @ y.T  # [N, M]
    return x2 - 2 * xy + y2  # [N, M]

def kernel(x, y, sigma=1.0):
    dists = pairwise_squared_dist(x, y)
    return torch.exp(-dists / (2 * sigma**2))

def kernel_mean_embedding(x, y, sigma=1.0):
    kxx = kernel(x, x, sigma).mean()
    kyy = kernel(y, y, sigma).mean()
    kxy = kernel(x, y, sigma).mean()
    return kxx + kyy - 2 * kxy

def custom_distance(x1, x2):
    return kernel_mean_embedding(x1, x2, sigma=1.0)
@torch.no_grad()
def simulator(theta):
    """
    theta: shape [4] (single parameter sample)
    returns: point cloud of shape [N_events, 6]
    """
    # Replace this with your actual DIS simulation logic
    # For now, we'll mock something realistic
    N = 100000

    def up(self, x):
        return self.Nu * (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return self.Nd * (x ** self.ad) * ((1 - x) ** self.bd)

    xs_p = torch.rand(N, device=self.device)
    sigma_p = 4 * up(xs_p) + down(xs_p)
    sigma_p = torch.nan_to_num(sigma_p, nan=0.0)

    xs_n = torch.rand(N, device=self.device)
    sigma_n = 4 * down(xs_n) + up(xs_n)
    sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

    x = torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()
    # x = torch.randn(N, 6) + theta[None, 0].item()  # use θ₁ as offset
    return x.float()

def simulator_batch(theta_batch):
    """
    theta_batch: shape [B, 4]
    returns: list of [N_events, 6] tensors, each as an observation
    """
    return [simulator(theta[i]) for i in range(theta_batch.shape[0])]

########################
# SBI PIPELINE
########################
if __name__ == "__main__":
    # Wrap simulator + prepare prior
    def wrapped_simulator(theta_batch):
        # return a list of tensors, which simulate_for_sbi will accept
        return simulator_batch(theta_batch)

    wrapped_simulator, wrapped_prior = prepare_for_sbi(wrapped_simulator, prior)

    # Simulate training data
    num_simulations = 500  # Keep this manageable
    theta, x = simulate_for_sbi(wrapped_simulator, wrapped_prior, num_simulations=num_simulations)

    # Define inference method with custom distance
    inference = MCABC(prior=wrapped_prior, simulator=wrapped_simulator, distance=custom_distance)

    # True parameter and observed data
    true_theta = torch.tensor([1.0, 2.0, 0.5, 4.0])
    x_o = simulator(true_theta)

    # Run inference
    posterior = inference(x_o)

    # Sample from posterior
    samples = posterior.sample((1000,))
    print("Posterior mean:", samples.mean(dim=0))
    print("Posterior std:", samples.std(dim=0))

def histogram_summary(x, nbins=32, range_min=-10, range_max=10):
    """
    Convert point cloud [N, D] to a fixed-size vector [D * nbins].
    - Bins are the same for every x.
    - Histogram is normalized per feature.
    """
    D = x.shape[1]
    summaries = []

    for d in range(D):
        # Use torch.histc for fast binning
        hist = torch.histc(x[:, d], bins=nbins, min=range_min, max=range_max)
        hist = hist / (hist.sum() + 1e-8)  # normalize to sum to 1
        summaries.append(hist)

    return torch.cat(summaries, dim=0)  # shape: [D * nbins]

def snpe_benchmark(simulator, prior, nbins=32):
    def sim_fn(theta):
        return histogram_summary(simulator(theta), nbins=nbins)

    # Wrap simulator and prior
    sim_fn_wrapped, prior_wrapped = prepare_for_sbi(sim_fn, prior)

    # Simulate training data
    theta, x = simulate_for_sbi(sim_fn_wrapped, prior_wrapped, num_simulations=1000)

    # Inference
    inference = SNPE(prior_wrapped)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)

    return posterior

def sliced_wasserstein(x1, x2, num_projections=50):
    """
    x1, x2: [N, D]
    Returns: average 1D Wasserstein distance
    """
    d = x1.shape[1]
    distances = []
    for _ in range(num_projections):
        proj = torch.randn(d)
        proj = proj / torch.norm(proj)
        x1_proj = x1 @ proj
        x2_proj = x2 @ proj
        w = wasserstein_distance(x1_proj.cpu().numpy(), x2_proj.cpu().numpy())
        distances.append(w)
    return np.mean(distances)

def wasserstein_distance_wrapper(x1, x2):
    return sliced_wasserstein(x1, x2)

def wasserstein_abc_benchmark(simulator, prior):
    # Wrap for sbi
    def sim_fn_batch(theta_batch):
        return [simulator(theta) for theta in theta_batch]

    sim_fn_wrapped, prior_wrapped = prepare_for_sbi(sim_fn_batch, prior)

    inference = MCABC(prior=prior_wrapped, simulator=sim_fn_wrapped, distance=wasserstein_distance_wrapper)

    true_theta = torch.tensor([1.0, 2.0, 0.5, 4.0])
    x_o = simulator(true_theta)

    posterior = inference(x_o)
    return posterior

if __name__ == "__main__":
    # Define prior
    prior = BoxUniform(low=torch.zeros(4), high=5 * torch.ones(4))

    # Define true observation
    true_theta = torch.tensor([1.0, 2.0, 0.5, 4.0])
    x_o = simulator(true_theta)

    ### 1. MCABC with kernel
    print("Running MCABC (kernel MMD)...")
    posterior_mmd = MCABC(prior, simulator, distance=custom_distance)(x_o)
    samples_mmd = posterior_mmd.sample((1000,))
    print("MMD Posterior mean:", samples_mmd.mean(0))

    ### 2. SNPE (histograms)
    print("Running SNPE (histograms)...")
    posterior_snpe = snpe_benchmark(simulator, prior)
    x_o_hist = histogram_summary(x_o)
    samples_snpe = posterior_snpe.sample((1000,), x=x_o_hist)
    print("SNPE Posterior mean:", samples_snpe.mean(0))

    ### 3. Wasserstein ABC
    print("Running Wasserstein ABC...")
    posterior_wasserstein = wasserstein_abc_benchmark(simulator, prior)
    samples_wass = posterior_wasserstein.sample((1000,))
    print("Wasserstein Posterior mean:", samples_wass.mean(0))