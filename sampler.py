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

# import tomography_toolkit_dev.envs.sampler_module as samplers
# import tomography_toolkit_dev.envs.theory_module as theories
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

tparams = torch.tensor([1])
torch.manual_seed(42)


def generate_synthetic_data(
    params, nevents, device=torch.device("cpu"), option="normal", mixes=2, mixed_dims=2
):  # Function for generating synthetic data
    with torch.autograd.set_grad_enabled(True):
        if option == "proxy_data" or option == "theory":
            params = torch.unsqueeze(params, 0).to(device)
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
        elif option == "jlab":
            mellin = MELLIN(npts=8)
            alphaS = ALPHAS()
            eweak = EWEAK()
            pdf = PDF(mellin, alphaS)
            pdf.update_params(params.data)
            idis = THEORY(mellin, pdf, alphaS, eweak)
            mceg = MCEG(idis, rs=140, tar="p", W2min=10, nx=30, nQ2=20)
            if nevents != 1:
                events = torch.tensor(mceg.gen_events(nevents)).to(device)
            else:
                events = (
                    torch.tensor(mceg.gen_events(100))[0, :].unsqueeze(0).to(device)
                )
    return events
