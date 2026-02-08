# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:16:33 2023

@author: Jean-Baptiste Bouvier

Simplest implementation of 1D Denoising Diffusion Probabilistic Models (DDPM).   
Built from Algorithms 1 and 2 of the paper:
"Denoising Diffusion Probabilistic Models".
"""

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


#%% Hyperparameters

### Noise scales
noise_scales = torch.tensor([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])#, 0.999])

### Initial data distribution
mu_data = 1
sigma_data = 0.01

dim = 1000 # dimension of data


#%% Neural Networks for the denoisers

### Class for single denoiser for a given step on the noise scale
class denoiser(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.input_size = 1
        self.net = nn.Sequential(nn.Linear(self.input_size, self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.input_size) )
        
    def forward(self, x):
        """Takes noised data and predict noise level."""
        return self.net(x)
    
### Class to gather all the denoisers, 1 for each step on the noise scale
class Denoisers():
    def __init__(self, width, noise_scales):
        self.noise_scales = noise_scales
        for id_noise_scale in range(len(noise_scales)):
            setattr(self, f"d_{id_noise_scale}", denoiser(width) )

denoisers = Denoisers(32, noise_scales)

#%% Training loop with iteration on the denoisers

for id_noise_scale in range(len(noise_scales)):
    
    ### Denoiser for this noise scale           
    den = getattr(denoisers, f"d_{id_noise_scale}")
    
    if id_noise_scale == 0: # first NN needs more training
        lr = 2e-3
        nb_epochs = 20000
    else:
        lr = 1e-5
        nb_epochs = 1000
        ### Warmstart weights to those of previous noise scale
        den.load_state_dict(getattr(denoisers, f"d_{id_noise_scale-1}").state_dict())
    
    alpha_bar = torch.prod(1 - noise_scales[:id_noise_scale+1])
    losses = np.zeros(nb_epochs)
    optimizer = torch.optim.SGD(den.parameters(), lr)
    
    for epoch in range(nb_epochs):
    
        x0 = torch.randn((dim,1))*sigma_data + mu_data # initial unnoised data
        eps = torch.randn_like(x0) # noise
        x_noised = np.sqrt(alpha_bar)*x0 + np.sqrt(1-alpha_bar)*eps # adding noise corresponding to noise_scale
        pred = den(x_noised) # prediction of the noise level        
        loss = torch.linalg.vector_norm(eps - pred) # difference between actual noise and prediction
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # update weights
        losses[epoch] = loss.detach().item()
        if loss.item() < 5: lr = 1e-6
        
    plt.title(f"Loss for noise scale {id_noise_scale}")
    plt.plot(np.arange(nb_epochs), losses)
    plt.show()
    

#%% Denoising process

def plot_sample(x, id_noise_scale):
    plt.title(f"Noised scale {id_noise_scale}")
    plt.scatter(np.arange(dim), x.numpy())
    plt.ylim([-3., 3.])
    plt.show()

### Sample from the most noised distribution
alpha_bar = torch.prod(1 - noise_scales)
x = torch.randn((dim,1))*torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar)
plot_sample(x, len(noise_scales))

print(f"Sample: mean {x.mean().item():.3f}  std {x.std().item():.3f}")

for id_noise_scale in range(len(noise_scales)-1, -1, -1):
    if id_noise_scale > 1:
        z = torch.randn_like(x)
    else:
        z = torch.zeros_like(x)
        
    alpha = 1 - noise_scales[id_noise_scale]
    alpha_bar = torch.prod(1 - noise_scales[:id_noise_scale+1])
    den = getattr(denoisers, f"d_{id_noise_scale}")
    # sigma_sq = noise_scales[id_noise_scale] # Two choices for sigma
    sigma_sq = noise_scales[id_noise_scale] * (1 - alpha_bar/alpha)/(1 - alpha_bar)
    with torch.no_grad():
        x = (x - (1-alpha)*den(x)/np.sqrt(1-alpha_bar) )/np.sqrt(alpha) + torch.sqrt(sigma_sq)*z
    plot_sample(x, id_noise_scale)
    print(f"Sample: mean {x.mean().item():.3f}  std {x.std().item():.3f}")
    
print(f"Goal:   mean {mu_data:.3f}  std {sigma_data:.3f}")