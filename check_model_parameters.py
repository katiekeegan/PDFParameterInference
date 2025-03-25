from models import *
import torch

Generator = ConvolutionalGAN(32, 33)
Discriminator = MLP()
SurrogatePhysics = SurrogatePhysicsModel(input_dim=33)

# Calculate the number of trainable parameters
trainable_params = sum(p.numel() for p in Generator.parameters() if p.requires_grad)
print('G: ',trainable_params)
# Calculate the number of trainable parameters
trainable_params = sum(p.numel() for p in Discriminator.parameters() if p.requires_grad)
print('D: ',trainable_params)
# Calculate the number of trainable parameters
trainable_params = sum(p.numel() for p in SurrogatePhysics.parameters() if p.requires_grad)
print('S: ',trainable_params)