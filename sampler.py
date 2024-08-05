from __future__ import print_function

import sys

# appending a path
sys.path.append('./theory/jamlib/')
from alphaS import ALPHAS
from eweak import EWEAK
from pdf import PDF
from mellin import MELLIN
from idis import THEORY
from mceg import MCEG
import argparse
import io
import tomography_toolkit_dev.envs.sampler_module as samplers
import tomography_toolkit_dev.envs.theory_module as theories
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
import torch.distributions as D
import torch.nn.functional as F
import torch.distributions as dist

from ddpm_functions import *

torch.autograd.set_detect_anomaly(True)

# Set true parameters
# if args.distribution == "normal":
#     mu = 1.6  # true mean of Gaussian
#     std = 0.8  # true std of Gaussian
#     tparams = torch.tensor([mu, std])
# elif args.distribution == "exp":
#     rate = 2  # true rate of exponential distribution
#     tparams = torch.tensor([rate])
# elif args.distribution == "mixture":
#     rate = 2  # true rate of exponential distribution
#     tparams = torch.tensor([rate])
# elif args.distribution == "2D":
#     rate = 2  # true rate of exponential distribution
tparams = torch.tensor([1])
torch.manual_seed(42)

class MixtureSameFamily(torch.nn.Module):
    def __init__(self, num_components, num_dimensions, num_mixtures):
        super(MixtureSameFamily, self).__init__()
        self.num_components = num_components
        self.num_dimensions = num_dimensions
        self.num_mixtures = num_mixtures

        # Parameters for the mixture distribution
        self.pi = torch.nn.Parameter(torch.ones(num_components) / num_components)

        # Parameters for component distributions
        self.mu = torch.nn.Parameter(torch.randn(num_components, num_mixtures, num_dimensions))
        self.sigma = torch.nn.Parameter(torch.ones(num_components, num_mixtures, num_dimensions))

    def forward(self):
        # Create mixture distribution
        mixture_distribution = torch.distributions.Categorical(self.pi)

        # Create component distributions
        component_distribution = torch.distributions.Normal(self.mu, self.sigma.exp())

        # Create mixture of mixtures of Gaussians distribution
        mixture = torch.distributions.MixtureSameFamily(mixture_distribution, component_distribution)

        return mixture

def generate_synthetic_data(
    params, nevents, device=torch.device("cpu"), option="normal", mixes=2, mixed_dims = 2
):  # Function for generating synthetic data
    with torch.autograd.set_grad_enabled(True):
        if option == "normal":
            mu = params[0]
            std = params[1]
            events = mu + std * torch.normal(0.0, 1.0, size=(nevents, 1), device=device)
            print(events.requires_grad)
        elif option == "exp":
            rate =params[0].item()#.float()
            # print(rate.requires_grad)
            # print(rate.item())
            if rate < 0:
                rate = 1e-8
            # exponential_dist = torch.distributions.Exponential(rate)
            # events = exponential_dist.sample(sample_shape = (nevents,1))
            events = torch.empty((nevents,1))
            events.exponential_(lambd=rate)
            events.requires_grad = True
            # u = torch.rand((nevents,1))
            # events = -1 / rate * torch.log(1 - u)
            # events.requires_grad_()
            # print('events')
            # print(events.size())
            # del exponential_dist
        elif option == "proxy_data" or option == "theory":
            params = torch.unsqueeze(params,0).to(device)
            # Define a config for the theory module:
            theory_cfg = {
                "parmin": [0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
                "parmax": [3.0, 1.0, 5.0, 3.0, 1.0, 5.0],
            }
            # Load the theory module, we need something to predict some data...
            proxy_theory = theories.make(
                "torch_proxy_theory_v0", config=theory_cfg, devices=device
            )
            # Load the sampler:
            sampler_cfg = {}
            inv_proxy_sampler = samplers.make(
                "torch_inv_proxy_sampler_v0", config=sampler_cfg, devices=device
            )
            sigma1, sigma2, norm1, norm2 = proxy_theory.forward(params)
            events, norm1, norm2 = inv_proxy_sampler.forward(
                sigma1, sigma2, norm1, norm2, nevents
            )
        # elif option == "jlab":
        #     events = tor
        elif option == "mixture":
            mix = torch.distributions.Categorical(
                torch.ones(
                    3,
                )
            )
            comp = torch.distributions.Normal(
                torch.tensor([-0.5, 1.5,3], dtype=torch.float32),
                torch.tensor([0.2, 0.4,0.3], dtype=torch.float32),
            )
            gmm = torch.distributions.MixtureSameFamily(mix, comp)
            events = gmm.sample((int(nevents), 1))
        elif option == "jlab":
            mellin = MELLIN(npts=8)
            alphaS = ALPHAS()
            eweak = EWEAK()
            pdf = PDF(mellin, alphaS)
            pdf.update_params(params.data)
            idis = THEORY(mellin, pdf, alphaS, eweak)
            mceg = MCEG(idis, rs=140, tar='p', W2min=10, nx=30, nQ2=20)
            if nevents != 1:
                events = torch.tensor(mceg.gen_events(nevents)).to(device)
            else:
                events = torch.tensor(mceg.gen_events(100))[0,:].unsqueeze(0).to(device)

        elif option == "multidimensional":
            torch.random.manual_seed(42)
            # Generate a random mean vector
            mean = torch.linspace(-10,10,4)

            # Covariance matrix (using identity matrix for simplicity)
            covariance = torch.eye(4)

            # Create a multivariate normal distribution
            multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance)

            # Generate samples
            events = multivariate_normal.sample((int(nevents), ))

        elif option == "mixed_multidimensional":
            # Define parameters for Gaussian mixture distributions
            means = torch.randn(4, 4)
            covariances = torch.rand(4, 4)
            weights = torch.rand(4)

            # Normalize weights to ensure they sum to 1
            weights /= torch.sum(weights)

            samples = []
            for _ in range(nevents):
                # Generate samples for each dimension
                dim_samples = torch.zeros(4)
                for dim in range(4):
                    # Sample a component index from the categorical distribution
                    component_index = torch.multinomial(weights, 1).item()
                    # Sample from the Gaussian distribution corresponding to the chosen component
                    sample = torch.normal(means[dim, component_index], covariances[dim, component_index])
                    dim_samples[dim] = sample
                samples.append(dim_samples)

            events = torch.stack(samples)

        elif option == "mixed_multidimensional_8D":
            p = 5
            n = 8
            means, variances, weights = unpack_params(params,p, n)

            samples = []

            for i in range(p):
                mean = means[i]
                var = variances[i]
                weight = weights[i]

                distribution = torch.distributions.MultivariateNormal(mean, var)
                component_samples = distribution.sample([int(nevents * weight.item())])
                samples.append(component_samples)
            events = torch.cat(samples)

        elif option == "mixed":
            # mix = torch.distributions.Categorical(torch.rand(1, 5))
            # comp = torch.distributions.Independent(torch.distributions.Normal(torch.randn(1, 5, 4), torch.rand(1, 5, 4)), 1)
            # gmm = torch.distributions.MixtureSameFamily(mix, comp)
            mix_params = torch.tensor([1.0, 0.5]).to(device) #equally mixed
            mix = torch.distributions.Categorical(mix_params)#.to(device)
            means = params.reshape(mixes, mixed_dims).to(device)
            # comp_1 = params[...,:2*mixes]
            # comp_2 = params[...,2*mixes:]
            # comp_1, comp_2 = params.chunk(2, dim=-1)
            # mask = comp_2 < 0  # Create a mask for negative entries
            # comp_2[mask] = 0.001  # Replace negative entries with 'a'
            comp = torch.distributions.Independent(torch.distributions.Normal(means, 0.1*torch.ones_like(means).to(device)), 1)
            gmm = torch.distributions.MixtureSameFamily(mix, comp)
            events = gmm.sample((int(nevents), )).to(device)
        elif option == "2D":
            data = np.load('idis_events_rejection.npy')
            events = data / np.max(data)
    return events

