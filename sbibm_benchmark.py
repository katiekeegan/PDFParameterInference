from sbi.inference import simulate_for_sbi, SNPE, MCABC
from sbi.utils.torchutils import BoxUniform
import torch
import numpy as np
from scipy.stats import wasserstein_distance

########################
# SIMULATOR DEFINITIONS
########################
@torch.no_grad()
def simulator(theta):
    N = 100000

    def up(x, theta):
        return 1 * (x ** theta[0]) * ((1 - x) ** theta[1])

    def down(x, theta):
        return 2 * (x ** theta[2]) * ((1 - x) ** theta[3])

    xs_p = torch.rand(N)
    sigma_p = 4 * up(xs_p, theta) + down(xs_p, theta)
    sigma_p = torch.nan_to_num(sigma_p, nan=0.0)

    xs_n = torch.rand(N)
    sigma_n = 4 * down(xs_n, theta) + up(xs_n, theta)
    sigma_n = torch.nan_to_num(sigma_n, nan=0.0)

    x = torch.cat([sigma_p.unsqueeze(0), sigma_n.unsqueeze(0)], dim=0).t()
    return x.float()

def simulator_batch(theta_batch):
    return [simulator(theta) for theta in theta_batch]

def simulator_batch_summary(theta_batch):
    """
    Returns [B, summary_dim] where each row is a feature vector (summary)
    """
    return torch.stack([histogram_summary(simulator(theta)) for theta in theta_batch])

def histogram_summary(x, nbins=32, range_min=-10, range_max=10):
    D = x.shape[1]
    summaries = []
    for d in range(D):
        hist = torch.histc(x[:, d], bins=nbins, min=range_min, max=range_max)
        hist = hist / (hist.sum() + 1e-8)
        summaries.append(hist)
    return torch.cat(summaries, dim=0)

def snpe_benchmark(simulator, param_prior, nbins=32):
    def sim_fn(theta_batch):
        return simulator_batch_summary(theta_batch)
    theta, x = simulate_for_sbi(sim_fn, param_prior, num_simulations=10000)
    inference = SNPE(param_prior)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference.build_posterior(density_estimator)

def wasserstein_abc_benchmark(simulator, param_prior):
    inference = MCABC(prior=param_prior, simulator=simulator_batch, distance=wasserstein_distance_wrapper)
    true_theta = torch.tensor([1.0, 2.0, 0.5, 1.0])
    x_o = simulator(true_theta)
    return inference(x_o, num_simulations=10000, quantile=0.01)

########################
# KERNEL DISTANCE
########################
def pairwise_squared_dist(x, y):
    x2 = x.pow(2).sum(dim=1, keepdim=True)
    y2 = y.pow(2).sum(dim=1, keepdim=True).T
    xy = x @ y.T
    return x2 - 2 * xy + y2

def kernel(x, y, sigma=1.0):
    return torch.exp(-pairwise_squared_dist(x, y) / (2 * sigma**2))

def kernel_mean_embedding(x, y, sigma=1.0):
    return kernel(x, x, sigma).mean() + kernel(y, y, sigma).mean() - 2 * kernel(x, y, sigma).mean()

def custom_distance(x1, x2):
    if x1.ndim == 1:
        x1 = x1.unsqueeze(0)
    if x2.ndim == 1:
        x2 = x2.unsqueeze(0)
    raw = kernel_mean_embedding(x1, x2, sigma=1.0)
    return (raw / (1.0 + raw)).item()  # return a Python float

def sliced_wasserstein(x1, x2, num_projections=50):
    if x1.ndim == 1:
        x1 = x1.unsqueeze(0)  # [1, D]
    if x2.ndim == 1:
        x2 = x2.unsqueeze(0)  # [1, D]

    d = x1.shape[1]
    distances = []
    for _ in range(num_projections):
        proj = torch.randn(d, device=x1.device)
        proj = proj / torch.norm(proj)

        x1_proj = (x1 @ proj).view(-1).cpu().numpy()  # always 1D
        x2_proj = (x2 @ proj).view(-1).cpu().numpy()

        if len(x1_proj) == 0 or len(x2_proj) == 0:
            distances.append(np.nan)
        else:
            w = wasserstein_distance(x1_proj, x2_proj)
            distances.append(w)

    return np.nanmean(distances)

def wasserstein_distance_vectorized(x1, x_batch):
    """
    x1: [N1, D] sample from the observation
    x_batch: [B, N2, D] batch of simulated samples
    returns: torch.tensor of shape [B] with scalar distances
    """
    distances = []
    for x2 in x_batch:
        dist = sliced_wasserstein(x1, x2)  # returns scalar
        distances.append(dist)
    return torch.tensor(distances)

def custom_distance_vectorized(x1, x_batch):
    """
    x1: [D] summary vector of observation (unbatched)
    x_batch: [B, D] batch of simulated summaries
    returns: torch.tensor of shape [B] with scalar distances
    """
    if x1.ndim == 1:
        x1 = x1.unsqueeze(0)  # [1, D]

    distances = []
    for x2 in x_batch:
        distances.append(custom_distance(x1.squeeze(0), x2))  # still returns float

    return torch.tensor(distances)  # [B]

def wasserstein_distance_wrapper(x1, x2):
    return sliced_wasserstein(x1, x2)

if __name__ == "__main__":
    # Avoid any previous name conflicts: define a clean variable
    prior_dist = BoxUniform(low=torch.zeros(4), high=5 * torch.ones(4))

    # True observation
    true_theta = torch.tensor([1.0, 0.5, 1.2, 0.5])
    x_o = simulator(true_theta)
    x_o_summary = histogram_summary(x_o)
    # Try this after simulation
    samples = simulator_batch_summary(prior_dist.sample((100,)))
    dists = [custom_distance(x_o_summary, s) for s in samples]
    print("Example distances:", dists[:10])
    print("Mean distance:", torch.tensor(dists).mean())
    print("Example distances:", dists[:10])
    print("Mean distance:", torch.tensor(dists).mean())
    ### 1. MCABC (MMD)
    print("Running MCABC (kernel MMD)...")
    samples_mmd = MCABC(
        prior=prior_dist,
        simulator=simulator_batch_summary,
        distance=custom_distance_vectorized  # should now accept summary vectors
    )(x_o_summary, num_simulations=10000, quantile=0.01)
    # samples_mmd = posterior_mmd#.sample((1000,))
    # Convert to numpy and save
    np.savetxt("samples_mmd.txt", samples_mmd.cpu().numpy())
    print("MMD Posterior median:", samples_mmd.median(0))

    ### 2. SNPE
    print("Running SNPE (histograms)...")
    posterior_snpe = snpe_benchmark(simulator, prior_dist)
    x_o_hist = histogram_summary(x_o)
    samples_snpe = posterior_snpe.sample((100,), x=x_o_hist)
    print("SNPE Posterior mean:", samples_snpe.mean(0))
    np.savetxt("samples_snpe.txt", samples_snpe.cpu().numpy())
    ### 3. Wasserstein ABC
    print("Running Wasserstein ABC...")
    # posterior_wasserstein = wasserstein_abc_benchmark(simulator_batch_summary, prior_dist)
    # samples_wass = posterior_wasserstein.sample((100,))
    samples_wass = MCABC(
        prior=prior_dist,
        simulator=simulator_batch_summary,
        distance=wasserstein_distance_vectorized  # should now accept summary vectors
    )(x_o_summary, num_simulations=10000, quantile=0.01)
    # samples_mmd = posterior_mmd#.sample((1000,))
    print("Wasserstein Posterior mean:", samples_wass.mean(0))
    np.savetxt("samples_wasserstein.txt", samples_wass.cpu().numpy())