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

lrs = [0.01]
G_steps = [1,5,10]
D_steps = [1,5,10]
S_steps = [1,5,10]
n_true_events = [1024,2048,10240]

device = torch.device("cpu")

np.random.seed(42)

tparams = np.array([-6.8000e-02,  1.0000e+01, -4.5000e+00,
  5.7700e+00, -7.1000e-01,
         3.4800e+00,  1.3400e+00,  2.3300e+01, -7.8000e-01,  4.8700e+00,
        -5.8000e+00,  4.7100e+01,  6.0000e-03, -4.1000e-01,  2.2500e+01,
         6.0000e-03, -4.1000e-01,  2.2500e+01,  3.1000e-02, -6.9000e-01,
         1.0000e+01,  8.1100e+01, -8.2200e+01,  2.6000e-02,  2.9000e-01,
         1.0100e+01, -3.7300e+00,  4.2700e+00, -7.8347e-01,  5.9491e+00,
         3.1853e-02, -4.7618e-01,  1.0000e+01])

theory_cfg = {
    "parmin": [ -2,   0, -10, -10,  -1,   0, -10, -10,  -1,   0, -10, -10, -10,
        -2,   0, -10,  -2,   0, -10,  -2,   0, -10, -10, -10,  -2,   0,
       -10, -10,  -2,   0, -10,  -2,   0],
    "parmax":[ 10,  10, -10, -10,  10,  10, -10, -10,  10,  10, -10, -10,  10,
        10,  10,  10,  10,  10,  10,  10,  10, -10, -10,  10,  10,  10,
       -10, -10,  10,  10,  10,  10,  10],
}

random_params = [np.array([
    random.uniform(min_val, max_val) 
    for min_val, max_val in zip(theory_cfg["parmin"], theory_cfg["parmax"])
]) for i in range(3)]

for i, (lr, G_step, D_step, S_step, n_true_event) in enumerate(itertools.product(lrs, G_steps, D_steps, S_steps, n_true_events)):
    print(i)
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
        + str(n_true_event)
    )

    Generator = ConvolutionalGAN(32, 33).to(device)
    try:
        # Load the saved state_dict into the model
        Generator.load_state_dict(torch.load(filename + "model.pt"))
    except FileNotFoundError:
        continue

    # Set the model to evaluation mode if you're using it for inference
    Generator.eval()

    z = torch.randn(32, 32).to(device)

    gen_params = Generator(z).mean(dim=0)

    gen_data = generate_synthetic_data(
            gen_params, n_true_event, device, option='jlab'
        ).float().squeeze().detach().numpy()

    random_data = [generate_synthetic_data(
                random_params[i], n_true_event, option='jlab', device=device
            ).float().squeeze().detach().numpy() for i in range(3)]

    data = generate_synthetic_data(
            tparams, n_true_event, device, option='jlab'
        ).float().squeeze().detach().numpy()

    gen_chi2_stat, gen_p_value =  chisquaremetric(gen_data, data)
    random_chi2_stats = []
    random_p_values = []
    for i in range(3):
        random_chi2_stat, random_p_value =  chisquaremetric(random_data[i], data)
        random_chi2_stats.append(random_chi2_stat)
        random_p_values.append(random_p_value)
    gen_emd = emd(gen_data, data)
    random_emd = np.mean([emd(random_data[i], data) for i in range(3)])
    G_error = np.linalg.norm(tparams - gen_params.detach().numpy())
    df_data = {
        'lr': [lr],
        'G_steps': [G_step],
        'D_steps': [D_step],
        'S_steps': [S_step],
        'n_true_events': [n_true_event],
        'gen_chi2_stat': [gen_chi2_stat],
        'gen_p_value': [gen_p_value],
        'random_chi2_stat': [np.mean(random_chi2_stats)],
        'random_p_value': [np.mean(random_p_values)],
        'gen_emd': [gen_emd],
        'random_emd': [random_emd],
        'G_error': [G_error]
    }
    df = pd.DataFrame(df_data)

    # df.to_csv('output.csv', mode='a', index=False, header=not pd.read_csv('output.csv').empty if 'output.csv' else True)

    # Check if the file exists
    if os.path.exists('output.csv'):
        # Append to the file
        df.to_csv('output.csv', mode='a', index=False, header=False)
    else:
        # Write to a new file (with header)
        df.to_csv('output.csv', mode='w', index=False, header=True)
