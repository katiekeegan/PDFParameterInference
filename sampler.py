from __future__ import print_function

import sys

# appending a path
sys.path.append("./theory/jamlib/")
from alphaS import ALPHAS
from eweak import EWEAK
from pdf import PDF
from mellin import MELLIN
from idis import THEORY
from mceg import MCEG
import numpy as np

from gan_functions import unpack_params

# import tomography_toolkit_dev.envs.sampler_module as samplers
# import tomography_toolkit_dev.envs.theory_module as theories
import torch
import torch.distributions as D

torch.autograd.set_detect_anomaly(True)

tparams = torch.tensor([1])
torch.manual_seed(42)

import torch
import torch.distributions as dist

class MultivariateMixtureModel:
    def __init__(self, means, covariances, weights):
        """
        Initialize the multivariate mixture model.

        :param means: A tensor of shape (n_components, n_features) representing the means of each component.
        :param covariances: A tensor of shape (n_components, n_features, n_features) representing the covariance matrix of each component.
        :param weights: A tensor of shape (n_components,) representing the mixture weights.
        """
        self.means = means
        self.covariances = covariances
        self.weights = weights

        # Create a categorical distribution for selecting components
        self.component_dist = dist.Categorical(self.weights)

        # Create a list of multivariate normal distributions for each component
        self.gaussians = [dist.MultivariateNormal(mean, cov) 
                          for mean, cov in zip(self.means, self.covariances)]

    def sample(self, num_samples):
        """
        Sample from the multivariate mixture model.

        :param num_samples: The number of samples to generate.
        :return: A tensor of shape (num_samples, n_features) representing the samples.
        """
        # Sample component indices
        component_indices = self.component_dist.sample((num_samples,))

        # For each component index, sample from the corresponding multivariate normal
        samples = torch.stack([self.gaussians[i.item()].sample() for i in component_indices])
        return samples

def generate_synthetic_data(
    params, nevents, device=torch.device("cpu"), option="jlab"
):  # Function for generating synthetic data
    with torch.autograd.set_grad_enabled(True):
        if option == "poisson":
            dist = torch.distributions.poisson.Poisson(params)
            events = dist.sample((nevents,1)).to(device)
        elif option == "mixture":
            p = 2
            n = 2
            means = unpack_params(params,p, n)
            # print(means)
            covariances = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], 
                            [[0.1, 0.0], [0.0, 0.1]]])
            weights = torch.tensor([0.5,0.5])
            # breakpoint()
            mixture_model = MultivariateMixtureModel(means, covariances, weights)
            events = mixture_model.sample(nevents).to(device)  # Generate 1000 samples
            # breakpoint()
            # events = torch.cat(samples).to(device)s
            # dist = D.multivariate_normal.MultivariateNormal(params[0:2].to(device), torch.diag(torch.ones(2)).to(device))
            # events = dist.sample((int(nevents),1)).to(device)
        elif option == "jlab":
            mellin = MELLIN(npts=8)
            alphaS = ALPHAS()
            eweak = EWEAK()
            pdf = PDF(mellin, alphaS)
            orig_params = [-6.80000000e-02,  1.00000000e+01, -4.50000000e+00,  5.77000000e+00,
            -7.10000000e-01,  3.48000000e+00,  1.34000000e+00,  2.33000000e+01,
            -7.80000000e-01,  4.87000000e+00, -5.80000000e+00,  4.71000000e+01,
                6.00000000e-03, -4.10000000e-01,  2.25000000e+01,  6.00000000e-03,
            -4.10000000e-01,  2.25000000e+01,  3.10000000e-02, -6.90000000e-01,
                1.00000000e+01,  8.11000000e+01, -8.22000000e+01,  2.60000000e-02,
                2.90000000e-01,  1.01000000e+01, -3.73000000e+00,  4.27000000e+00,
            -7.83472900e-01,  5.94912366e+00,  3.18530400e-02, -4.76183100e-01,
                1.00000000e+01]
            new_params = orig_params # initialize new_params
            # param_list = pdf.get_current_par_array().tolist()
            # for i in [-0.71,3.48,1.34,23.3]:
            #     print(i)
            #     print(param_list.index(i))
            # breakpoint()
            # KK: best way I could think of right now was updating the parameters by figuring out the corresponding indices
            # in the above array to pdf.params['uv1'][1:-1]. I needed to do this anyway to deal with the min and max penalties. 
            # there are probably more efficient ways to doing this. 

            # index(['uv1'][1]) = 4
            # index(['uv1'][2]) = 5
            # index(['uv1'][3]) = 6
            # index(['uv1'][4]) = 7
            # breakpoint()
            pdf.params['uv1'][1:-1] = params.data.tolist()
            # for i in [4,5,6,7]:
            #     new_params[i] = params.data[i-4].tolist()
            # pdf.update_params(new_params)
            idis = THEORY(mellin, pdf, alphaS, eweak)
            mceg = MCEG(idis, rs=140, tar="p", W2min=10, nx=30, nQ2=20)
            if nevents != 1:
                events = torch.tensor(mceg.gen_events(nevents + 1000)).to(device)
                random_indices = np.random.choice(events.size(0), size=nevents, replace=False)
                events = events[random_indices, :].unsqueeze(0)
            else:
                events = (
                    torch.tensor(mceg.gen_events(100)).to(device)
                )
                events = events[np.random.randint(0,events.size(0)), :].unsqueeze(0)
    return events
