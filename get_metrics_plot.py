import numpy as np
import matplotlib.pyplot as plt
from gan_functions import *
import pandas as pd
import torch
from models import ConvolutionalGAN
import random
from sampler import *
import os
import itertools
from plotting import *

lrs = [0.01]
G_steps = [1,5,10]
D_steps = [1,5,10]
S_steps = [1,5,10]

df = pd.read_csv('output.csv')

for i, (lr, G_step, D_step, S_step) in enumerate(itertools.product(lrs, G_steps, D_steps, S_steps)):
    plot_metrics_wrt_data(df, lr, G_step, D_step, S_step)

    filename = (
        "lr"
        + str(lr)
        + "noisedim32"
        + "G"
        + str(G_step)
        + "D"
        + str(D_step)
        + "S"
        + str(S_step)
        + "n"
        + str(1024)
    )

    if not os.path.exists(filename + "model.pt"):
        continue
    plt.clf()
    plot_metrics_wrt_data(df, lr, G_step, D_step, S_step)