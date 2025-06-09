# Benchmark your parton distribution inference problem with SBIBM
# using multiple state-of-the-art SBI algorithms

import sbibm
from sbibm.tasks.task import Task
from sbibm.algorithms import snle,snpe,snre, rej_abc
from sbibm.metrics import c2st
import torch
import os
from torch.distributions import Uniform
from typing import Callable, List, Tuple, Optional

class SimplifiedDIS:
    def __init__(self, device=None, smear=False, smear_std=0.05, num_samples=1):
        self.device = device or torch.device("cpu")
        self.smear = smear
        self.smear_std = smear_std
        self.num_samples = num_samples

    def init(self, params):
        self.au, self.bu, self.ad, self.bd = [p.to(self.device) for p in params]

    def up(self, x):
        return (x ** self.au) * ((1 - x) ** self.bu)

    def down(self, x):
        return (x ** self.ad) * ((1 - x) ** self.bd)

    def __call__(self, params, num_samples=1):
        if params.ndim == 1:
            params = params.unsqueeze(0)  # shape (1, dim)
        out = []
        for p in params:
            self.init(p)
            eps = 1e-6
            rand = lambda: torch.clamp(torch.rand(num_samples, device=self.device), min=eps, max=1 - eps)
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
        self.prior = torch.distributions.Uniform(torch.zeros(4), torch.ones(4) * 5.0)

    def get_simulator(self, max_calls= None):
        sim = SimplifiedDIS(torch.device("cpu"))
        def simulator(theta):
            return sim(theta)
        return simulator

    def get_prior_dist(self):
        return self.prior

    def get_prior(self):
        return self.prior

    def get_observation(self, seed, num_samples=1):
        # torch.manual_seed(seed)
        theta = self.get_prior().sample()
        x = self.get_simulator()(theta)
        return x.squeeze(), theta


# # === Task and Output Setup ===
task = PDFTask()

from sbi.inference.abc.mcabc import MCABC as RejectionABC

x_obs, true_theta = task.get_observation(0)
x_obs = x_obs.view(1, -1).float()

simulator = task.get_simulator()
prior = task.get_prior()

# abc = RejectionABC(prior=prior, simulator=simulator)
# posterior = abc(x_o=x_obs, num_simulations=1000000,quantile=0.01)
# print(posterior)
# print(f"TRUE: {true_theta}")

from sbi.inference.base import infer
# prior = [
#                             Uniform(torch.zeros(1), 5 * torch.ones(1)),
#                             Uniform(torch.zeros(1), 5 * torch.ones(1)),
#                             Uniform(torch.zeros(1), 5 * torch.ones(1)),
#                             Uniform(torch.zeros(1), 5 * torch.ones(1))
#                         ]
# posterior = infer(simulator, prior, "SNPE", num_simulations=100000)
# samples = posterior.sample((100,),x=x_obs)

from sbi.inference import SNPE_A

prior = torch.distributions.Uniform(torch.zeros(4), torch.ones(4) * 5.0)
num_sims=100000
# inference = SNPE_A(prior)

# SNRE A

# proposal = prior
# from sbi.inference import SNRE_A

# inference = SNRE_A(prior)
# theta = prior.sample((num_sims,))
# x = simulator(theta)
# _ = inference.append_simulations(theta, x).train()
# posterior = inference.build_posterior().set_default_x(x_o)

# from sbi.inference import simulate_for_sbi, prepare_for_sbi, MCABC
# import torch

# # Define your prior and simulator
# # prior = ...  # Some sbi-compatible torch distribution
# # simulator = ...  # A simulator that takes theta and returns x
# # Simulate data
# num_sims = 100000
# theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_sims)
# prior = [
#                             Uniform(torch.zeros(1), 5 * torch.ones(1)),
#                             Uniform(torch.zeros(1), 5 * torch.ones(1)),
#                             Uniform(torch.zeros(1), 5 * torch.ones(1)),
#                             Uniform(torch.zeros(1), 5 * torch.ones(1))
#                         ]
# # Prepare for inference (needed to wrap the simulator correctly)
# simulator, prior = prepare_for_sbi(simulator, prior)

# x_o = simulator(true_theta)  # your observed data

# # Run ABC inference
# inference = MCABC(prior=prior, simulator=simulator, distance="l2")

# # Get posterior conditioned on observation (uses internal sampling)
# samples = inference(x_o, num_simulations=10000000, eps=0.1)

# print(f"TRUE: {true_theta}")
# print(f"GENERATED: {samples.median(dim=0)}")

def rejection_abc_iid(
    simulator: Callable,
    prior: List[torch.distributions.Distribution],
    x_obs_list: List[torch.Tensor],  # List of observed datasets
    num_sims: int = 10000,
    eps: float = 0.1,
    distance_fn: Callable = lambda x, y: torch.norm(x - y, p=2)
) -> torch.Tensor:
    theta_samples = torch.stack([p.sample((num_sims,)) for p in prior], dim=1)
    accepted = []
    
    for theta in theta_samples:
        # Simulate one dataset per theta (same size as x_obs_list)
        x_sim_list = [simulator(theta) for _ in range(len(x_obs_list))]
        
        # Check distance for all observations
        if all(distance_fn(x_sim, x_obs) <= eps for x_sim, x_obs in zip(x_sim_list, x_obs_list)):
            accepted.append(theta)
    
    return torch.stack(accepted) if accepted else torch.zeros(0, len(prior))

def rejection_abc_iid(
    simulator: Callable,
    prior: List[torch.distributions.Distribution],
    x_obs_list: List[torch.Tensor],  # List of observed datasets
    num_sims: int = 10000,
    eps: float = 0.1,
    distance_fn: Callable = lambda x, y: torch.norm(x - y, p=2)
) -> torch.Tensor:
    theta_samples = torch.stack([p.sample((num_sims,)) for p in prior], dim=1)
    accepted = []
    
    for theta in theta_samples:
        # Simulate one dataset per theta (same size as x_obs_list)
        x_sim_list = [simulator(theta) for _ in range(len(x_obs_list))]
        
        # Check distance for all observations
        if all(distance_fn(x_sim, x_obs) <= eps for x_sim, x_obs in zip(x_sim_list, x_obs_list)):
            accepted.append(theta)
    
    return torch.stack(accepted) if accepted else torch.zeros(0, len(prior))

def hierarchical_abc(
    simulator: Callable,  # Simulator takes (theta, phi_i)
    prior_theta: List[torch.distributions.Distribution],
    prior_phi: Callable,  # Function of theta, e.g., lambda theta: Normal(theta, 1)
    x_obs_list: List[torch.Tensor],
    num_sims: int = 10000,
    eps: float = 0.1
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    theta_samples = torch.stack([p.sample((num_sims,)) for p in prior_theta], dim=1)
    accepted_theta = []
    accepted_phi = [[] for _ in x_obs_list]
    
    for theta in theta_samples:
        phi_list = [prior_phi(theta).sample() for _ in x_obs_list]
        x_sim_list = [simulator(theta, phi) for phi in phi_list]
        
        if all(torch.norm(x_sim - x_obs) <= eps for x_sim, x_obs in zip(x_sim_list, x_obs_list)):
            accepted_theta.append(theta)
            for i, phi in enumerate(phi_list):
                accepted_phi[i].append(phi)
    
    return torch.stack(accepted_theta), [torch.stack(phi) for phi in accepted_phi]

def pseudo_marginal_mcmc(
    simulator: Callable,
    prior: List[torch.distributions.Distribution],
    x_obs_list: List[torch.Tensor],
    num_mcmc_steps: int = 10000,
    num_sims_per_step: int = 100,  # Simulations per theta
    eps: float = 0.1
) -> torch.Tensor:
    theta_current = torch.stack([p.sample() for p in prior])
    log_prior_current = sum([p.log_prob(theta_current[i]).item() for i, p in enumerate(prior)])
    
    samples = []
    for _ in range(num_mcmc_steps):
        theta_proposed = theta_current + 0.1 * torch.randn_like(theta_current)
        log_prior_proposed = sum([p.log_prob(theta_proposed[i]).item() for i, p in enumerate(prior)])
        
        # Approximate likelihood for all x_obs
        log_likelihood = 0.0
        for x_obs in x_obs_list:
            x_sim = torch.stack([simulator(theta_proposed) for _ in range(num_sims_per_step)])
            log_likelihood += torch.log(torch.mean(torch.norm(x_sim - x_obs, dim=1) <= eps) + 1e-6)
        
        # Accept/reject
        if torch.rand(1) < torch.exp(log_prior_proposed + log_likelihood - log_prior_current):
            theta_current = theta_proposed
            log_prior_current = log_prior_proposed
        
        samples.append(theta_current.clone())
    return torch.stack(samples)

# pseudo_marginal_mcmc

# x_obs_list = [task.get_observation(i)[0] for i in range(3)]  # Get 3 observations
# x_obs, true_theta = task.get_observation(1, num_samples=100000)
true_theta = task.get_prior().sample()
simulator =  SimplifiedDIS(torch.device("cpu"))
x_obs_list = simulator(true_theta, num_samples=100000)

def rejection_abc_multi(task, x_obs_list, num_sims=1000000, eps=0.1):
    simulator = task.get_simulator()
    prior = task.get_prior()
    
    theta_samples = prior.sample((num_sims,))
    accepted = []
    
    for theta in theta_samples:
        # Simulate one dataset per theta
        x_sim = simulator(theta.unsqueeze(0)).squeeze(0)
        
        # Check distance for all observations
        if all(torch.norm(x_sim - x_obs) <= eps for x_obs in x_obs_list):
            print("Accepted!")
            accepted.append(theta)
    
    return torch.stack(accepted) if accepted else torch.zeros(0, task.dim_parameters)

def distance_summary(x1, x2):
    """Compare means and variances of sigma_p/sigma_n ratios"""
    ratio1 = x1[..., 0] / x1[..., 1]  # sigma_p / sigma_n for x1
    ratio2 = x2[..., 0] / x2[..., 1]  # sigma_p / sigma_n for x2
    return torch.abs(ratio1.mean() - ratio2.mean()) + torch.abs(ratio1.var() - ratio2.var())

def rejection_abc_summary(task, x_obs_list, num_sims=10000, eps=0.5):
    theta_samples = task.get_prior().sample((num_sims,))
    x_obs_stack = torch.stack(x_obs_list)
    accepted = []
    
    for theta in theta_samples:
        x_sim = task.get_simulator()(theta.unsqueeze(0)).squeeze(0).repeat(len(x_obs_list), 1)
        if distance_summary(x_sim, x_obs_stack) <= eps:
            accepted.append(theta)
    
    return torch.stack(accepted)

def rejection_abc_multi_relaxed(task, x_obs_list, num_sims=10000, eps=0.5, min_matches=1):
    theta_samples = task.get_prior().sample((num_sims,))
    accepted = []
    
    for theta in theta_samples:
        x_sim = task.get_simulator()(theta.unsqueeze(0)).squeeze(0)
        matches = sum(torch.norm(x_sim - x_obs) <= eps for x_obs in x_obs_list)
        if matches >= min_matches:  # Accept if at least 'min_matches' observations agree
            print("Accepted!")
            accepted.append(theta)
    
    return torch.stack(accepted) if accepted else torch.zeros(0, task.dim_parameters)


def hierarchical_abc(task, x_obs_list, num_sims=10000, eps=0.2):
    theta_samples = task.get_prior().sample((num_sims,))
    accepted = []
    
    for theta in theta_samples:
        # Simulate small perturbations around theta
        theta_perturbed = theta + 0.1 * torch.randn(len(x_obs_list), task.dim_parameters)
        x_sim = task.get_simulator()(theta_perturbed)
        
        # Check if all perturbed simulations match observations
        if all(torch.norm(x_sim[i] - x_obs_list[i]) <= eps for i in range(len(x_obs_list))):
            accepted.append(theta)
    
    return torch.stack(accepted)

# Usage: Accept if at least 1/3 observations match
samples = rejection_abc_multi_relaxed(task, x_obs_list, min_matches=100)

# multi_samples = rejection_abc_multi(task, x_obs_list)
print(f"Multi-observation accepted: {len(samples)} samples")