# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:53:02 2023
@author: Jean-Baptiste Bouvier

Simple implementation of 2D Diffusion Model
trained to remove one noise scale at a time.
"""

import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

#%% Hyperparameters
data_size = 2

### Initial data distribution
mu_data = torch.tensor([-1., 1.])
sigma_data = torch.diag(torch.tensor([0.01, 0.01]))

### Training
lr = 1e-3
batch_size = 32 
nb_epochs = 1000

### Noise scales
sigma_min = torch.min(torch.diag(sigma_data)).item()/10
sigma_max = 1.
N = 10 # number of noise scales

### Log-normal distribution of noise scales
rho = 7 # from "Elucidating the Design Space of Diffusion-Based Generative Models"
Sigmas = ( sigma_max**(1/rho) + torch.arange(N)*(sigma_min**(1/rho) - sigma_max**(1/rho))/(N-1) )**rho
Sigmas = torch.cat((Sigmas, torch.tensor([0.])), dim=0)

### Denoising
test_size = 2000

#%% Neural Networks for the denoisers

class Denoiser(nn.Module):
    def __init__(self, width, data_size):
        super().__init__()
        self.data_size = data_size
        self.net = nn.Sequential(nn.Linear(data_size+1, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, data_size) )
        
    def forward(self, x, sigma):
        s = sigma*torch.ones((x.shape[0], 1))
        return self.net( torch.cat((x, s), dim=1) )

denoiser = Denoiser(32, data_size)

#%% Training loop

losses = np.zeros(nb_epochs)
optimizer = torch.optim.SGD(denoiser.parameters(), lr)

for epoch in range(nb_epochs):
    
    id_sigma = random.randint(0, N-1) 
    sigma = Sigmas[id_sigma] # random noise level
    y = torch.randn((batch_size, data_size)) @ sigma_data + mu_data # initial unnoised data
    n = torch.randn_like(y)*sigma # noise
    
    pred = denoiser(y + n, sigma) # prediction of one noise level
    goal = n*Sigmas[id_sigma+1]/sigma # objective of the prediction
    loss = torch.sum( (pred - goal)**2, dim=(0,1) ) # trained to remove one level of noise
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    losses[epoch] = loss.detach().item()
    
plt.title("Training loss")
plt.plot(np.arange(nb_epochs), losses)
plt.show()
   

#%% Denoising process

x = torch.randn((test_size, data_size))*sigma_max # sample from the most noised distribution
plt.title("Noise scale 0")
plt.scatter(x[:,0], x[:,1], s=10)
plt.xlim([-3*sigma_max, 3*sigma_max])
plt.ylim([-3*sigma_max, 3*sigma_max])
plt.show()

print(f"Noised sample:   mean [{x[:,0].mean().item():.3f}, {x[:,1].mean().item():.3f}]  std [{x[:,0].std().item():.3f}, {x[:,1].std().item():.3f}]")
Mean,    Std    = np.zeros((N+1, data_size)),   np.zeros((N+1, data_size))
for j in range(data_size):
    Mean[0,j], Std[0,j] = x[:,j].mean().item(), x[:,j].std().item()

for i in range(N):
    with torch.no_grad():
        x -= denoiser(x, Sigmas[i]) # remove one level of noise
        
    plt.title(f"Noise scale {i+1}")
    plt.scatter(x[:,0], x[:,1], s=10)
    plt.xlim([-3*sigma_max, 3*sigma_max])
    plt.ylim([-3*sigma_max, 3*sigma_max])
    plt.show()
    
    for j in range(data_size):
        Mean[i+1, j], Std[i+1, j] = x[:,j].mean().item(), x[:,j].std().item()

print(f"Denoised sample:   mean [{x[:,0].mean().item():.3f}, {x[:,1].mean().item():.3f}]  std [{x[:,0].std().item():.3f}, {x[:,1].std().item():.3f}]")
print(f"Denoising goal:    mean [{mu_data[0]:.3f}, {mu_data[1]:.3f}]  std [{sigma_data[0,0]:.3f}, {sigma_data[1,1]:.3f}]")

plt.title("Mean of the denoising process")
for j in range(data_size):
    plt.plot(np.arange(N+1), Mean[:,j], label=f"x_{j}")
    plt.plot(np.array([0, N]), np.array([mu_data[j].item(), mu_data[j].item()]), label=f"data_{j}")
plt.xlabel("noise scales")
plt.legend()
plt.show()

plt.title("STD of the denoising process")
for j in range(data_size):
    plt.plot(np.arange(N+1), Std[:,j], label=f"x_{j}")
    plt.plot(np.array([0, N]), np.array([sigma_data[j,j].item(), sigma_data[j,j].item()]), label=f"data_{j}")
plt.xlabel("noise scales")
plt.legend()
plt.show()