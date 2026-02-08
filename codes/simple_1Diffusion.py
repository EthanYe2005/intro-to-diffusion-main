# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 08:50:21 2023
@author: Jean-Baptiste Bouvier

Simple implementation of 1D Diffusion Model
trained to remove one noise scale at a time.
"""

import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

#%% Hyperparameters

### Initial data distribution
mu_data = 1.
sigma_data = 0.01

### Training
lr = 1e-3
batch_size = 32 
nb_epochs = 1000

### Noise scales
sigma_min = sigma_data/10
sigma_max = 2.
N = 10 # number of noise scales

### Log-normal distribution of noise scales
rho = 7 # from "Elucidating the Design Space of Diffusion-Based Generative Models"
Sigmas = ( sigma_max**(1/rho) + torch.arange(N)*(sigma_min**(1/rho) - sigma_max**(1/rho))/(N-1) )**rho
print(f"噪声刻度表: {Sigmas}")
Sigmas = torch.cat((Sigmas, torch.tensor([0.])), dim=0)
print(f"噪声刻度表: {Sigmas}")
# 数值从 2.0 迅速下降，最后慢慢减小到 0。这代表了 11 个不同的“模糊程度”。

### Denoising
test_size = 1000

#%% Neural Networks for the denoisers

class Denoiser(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1+1, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, 1) )
        
    def forward(self, x, sigma):
        s = sigma*torch.ones_like(x)
        return self.net( torch.cat((x, s), dim=1) )

denoiser = Denoiser(32)

#%% Training loop

losses = np.zeros(nb_epochs)
optimizer = torch.optim.SGD(denoiser.parameters(), lr)

for epoch in range(nb_epochs):
    
    id_sigma = random.randint(0, N-1) 
    sigma = Sigmas[id_sigma] # random noise level
    y = torch.randn((batch_size,1))*sigma_data + mu_data # initial unnoised data
    n = torch.randn_like(y)*sigma # noise
    
    pred = denoiser(y + n, sigma) # prediction of the noise level    
    loss = torch.sum( (pred - n*Sigmas[id_sigma+1]/sigma )**2 ) # trained to remove one level of noise
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    losses[epoch] = loss.detach().item()
    
plt.title("Training loss")
plt.plot(np.arange(nb_epochs), losses)
plt.show()
   

#%% Denoising process

x = torch.randn((test_size, 1))*sigma_max # sample from the most noised distribution
print(f"Noised sample:   mean {x.mean().item():.3f}  std {x.std().item():.3f}")
Mean,    Std    = np.zeros(N+1),   np.zeros(N+1)
Mean[0], Std[0] = x.mean().item(), x.std().item()

for i in range(N):
    with torch.no_grad():
        x -= denoiser(x, Sigmas[i]) # remove one level of noise
    Mean[i+1], Std[i+1] = x.mean().item(), x.std().item()
   
print(f"Denoised sample: mean {x.mean().item():.3f}  std {x.std().item():.3f}")
print(f"Denoising goal:  mean {mu_data:.3f}  std {sigma_data:.3f}")

plt.title("Denoising process")
plt.plot(np.arange(N+1), Mean, label="mean")
plt.plot(np.array([0, N]), np.array([mu_data, mu_data]), label="mean data")
plt.plot(np.arange(N+1), Std, label="std")
plt.plot(np.array([0, N]), np.array([sigma_data, sigma_data]), label="std data")
plt.xlabel("noise scales")
plt.legend()
plt.show()