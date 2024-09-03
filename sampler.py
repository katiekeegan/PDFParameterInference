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

torch.autograd.set_detect_anomaly(True)

tparams = torch.tensor([1])
torch.manual_seed(42)


def generate_synthetic_data(
    params, nevents, device=torch.device("cpu"), option="jlab"
):  # Function for generating synthetic data
    with torch.autograd.set_grad_enabled(True):
        if option == "mixture":
            dist = torch.distributions.multivariate_normal.MultivariateNormal(params[0:2].to(device), torch.diag(torch.ones(2)).to(device))
            events = dist.sample((int(nevents),1)).to(device)
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
