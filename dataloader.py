import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from simulator import *

class UniformDataset(data.Dataset):
    def __init__(self, param_ranges, nevents=1, total_dataset_size = 100, device=None):
        self.param_ranges = torch.tensor(param_ranges, dtype=torch.float)
        self.nevents = nevents
        self.device = device
        self.total_dataset_size = total_dataset_size
        self.samples, self.param_samples = self._generate_data()
    def _generate_data(self):
        samples_total = []
        param_samples = torch.rand(self.total_dataset_size, len(self.param_ranges)) * (self.param_ranges[:, 1] - self.param_ranges[:, 0]) + self.param_ranges[:, 0]
        for i in range(0,self.total_dataset_size):
            simulator = SimplifiedDIS(device=self.device)
            # breakpoint()
            param = param_samples[i]
            samples = simulator.sample(param, self.nevents).to(self.device)
            samples_total.append(samples.unsqueeze(0))
        # samples = torch.cat([simulator.sample(param, self.nevents).unsqueeze(0) for param in param_samples], dim=0)
        samples_total = torch.cat(samples_total, dim=0)
        return samples_total, param_samples

    def __len__(self):
        return self.total_dataset_size

    def __getitem__(self, idx):
        return self.samples[idx], self.param_samples[idx]