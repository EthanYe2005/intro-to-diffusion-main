# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:03:31 2023
@author: Jean-Baptiste Bouvier

Simple implementation of 2D Diffusion Model with multimodality
trained to remove one noise scale at a time.
Initial data comes from 4 Gaussian with different means but same std
"""

import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

#%% Hyperparameters

### Initial data distribution
mu_data = torch.tensor([[-1., 1.], [1., -1.], [-1., -1.], [1., 1.]]) # center of each data spike
sigma_data = 0.01* torch.ones_like(mu_data)

data_size = mu_data.shape[1]
multimodality = mu_data.shape[0]

### Training
lr = 1e-3
batch_size = 32 # per modality
nb_epochs = 3000

### Noise scales
sigma_min = torch.min(sigma_data).item()/10
sigma_max = 1.
N = 20 # number of noise scales

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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

for epoch in range(nb_epochs):
    
    id_sigma = random.randint(0, N-1) 
    sigma = Sigmas[id_sigma] # random noise level
    y = torch.randn((batch_size, data_size)) * sigma_data[0,:] + mu_data[0,:] # initial unnoised data
    for modality in range(1, multimodality):
        y = torch.cat((y, torch.randn((batch_size, data_size))*sigma_data[modality,:] + mu_data[modality,:]), dim=0)
    n = torch.randn_like(y)*sigma # noise
    
    pred = denoiser(y + n, sigma) # prediction of one noise level
    goal = n*Sigmas[id_sigma+1]/sigma # objective of the prediction
    loss = torch.sum( (pred - goal)**2, dim=(0,1) ) # trained to remove one level of noise
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses[epoch] = loss.detach().item()
  
print(f"Average loss last 1000: {losses[-1000:].mean():.3f}")
plt.title("Training loss")
plt.plot(np.arange(nb_epochs), losses)
plt.show()
   

#%% Denoising process

def plot_sample(x, i):
    plt.title(f"N = {i}   sigma = {Sigmas[i]:.3f}")
    plt.scatter(x[:,0], x[:,1], s=10)
    plt.xlim([-3*sigma_max, 3*sigma_max])
    plt.ylim([-3*sigma_max, 3*sigma_max])
    plt.show()

def convergence_measure(x):
    """Calculates the sum of the distance of each point in x to the closest reference."""
    distance = torch.zeros((x.shape[0], multimodality))
    for mod in range(multimodality):
        distance[:,mod] = torch.sum( (x - mu_data[mod,:])**2, dim=1)
    d = distance.amin(dim=1).sum(dim=0)/x.shape[0]
    print(f"Average distance to target {d.item():.3f}")

x = torch.randn((test_size, data_size))*sigma_max # sample from the most noised distribution
plot_sample(x, 0)
convergence_measure(x)

for i in range(N):
    with torch.no_grad():
        x -= denoiser(x, Sigmas[i]) # remove one level of noise
    plot_sample(x, i+1)
    convergence_measure(x)
    
