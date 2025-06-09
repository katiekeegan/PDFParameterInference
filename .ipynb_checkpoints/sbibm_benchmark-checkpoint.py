# Benchmark your parton distribution inference problem with SBIBM
# using multiple state-of-the-art SBI algorithms

import sbibm
from sbibm.tasks.task import Task
from sbibm.algorithms import snle,snpe,snre
from sbibm.algorithms import get_algorithm
from sbibm.metrics import c2st
import torch
import os

class SimplifiedDIS:
    def __init__(self, device=None, smear=False, smear_std=0.05):
        self.device = device or torch.device("cpu")
        self.smear = smear
        self.smear_std = smear_std

    def init(self, params):
        self.au, self.bu, self.ad, self.bd = [p.to(self.device) for p in params]

    def up(self, x):
        return (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return (x ** self.ad) * ((1 - x) ** self.bd)

    def __call__(self, params):
        if params.ndim == 1:
            params = params.unsqueeze(0)  # shape (1, dim)
        out = []
        for p in params:
            self.init(p)
            eps = 1e-6
            rand = lambda: torch.clamp(torch.rand(1, device=self.device), min=eps, max=1 - eps)
            smear_noise = lambda s: s + torch.randn_like(s) * (self.smear_std * s) if self.smear else s
            xs_p, xs_n = rand(), rand()
            sigma_p = smear_noise(4 * self.up(xs_p) + self.down(xs_p))
            sigma_n = smear_noise(4 * self.down(xs_n) + self.up(xs_n))
            out.append(torch.stack([sigma_p, sigma_n], dim=-1).squeeze(0))
        return torch.stack(out)

# === Define Your Task ===
class PDFTask(Task):
    def __init__(self, name="pdf_sim", num_observation=1):
        super().__init__(
            name=name,
            dim_data=2,
            dim_parameters=4,
            num_observations=num_observation,
            num_posterior_samples=1000,
            num_simulations=1000,
            path="/tmp"  # or another directory you have write access to
        )

    def get_simulator(self):
        sim = SimplifiedDIS(torch.device("cpu"))
        def simulator(theta):
            return sim(theta)
        return simulator

    def get_prior(self):
        return torch.distributions.Uniform(torch.zeros(4), torch.ones(4) * 5.0)

    def get_observation(self, seed):
        torch.manual_seed(seed)
        theta = self.get_prior().sample()
        x = self.get_simulator()(theta)
        return x, theta

# === Task and Output Setup ===
task = PDFTask()
seed = 0
x_obs, true_theta = task.get_observation(seed)

results = {}

# === Benchmark using SBIBM algorithms ===
algorithm_names = ["snpe", "snl", "snre"]

for algo_name in algorithm_names:
    print(f"\nRunning {algo_name.upper()}...")
    run_algorithm = get_algorithm(algo_name)
    result = run_algorithm(
        task=task,
        num_simulations=1000,
        num_observation=seed,
        num_posterior_samples=1000,
        seed=seed
    )
    # === Sample posterior
    posterior = result.posterior
    observation = result.observation
    samples = posterior.sample((1000,), x=observation)

    
    # Evaluate metrics
    results[algo_name] = {
        "c2st": c2st(samples, true_theta)
    }

# === Print All Metrics ===
for algo_name, metrics in results.items():
    print(f"\nResults for {algo_name.upper()}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
