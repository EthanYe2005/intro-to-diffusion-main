# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:12:53 2023
@author: Jean-Baptiste Bouvier

Implementation of multimodal 2D Denoising Diffusion Probabilistic Models (DDPM).   
Built from Algorithms 1 and 2 of the papers:
"Denoising Diffusion Probabilistic Models".
Single neural network working for all noise scales
The data comes from 4 Gaussians with different means
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
multimodality = mu_data.shape[0]
data_size = mu_data.shape[1]

### Training
nb_epochs = 10000
width = 32
lr = 2e-3
batch_size = 32 # per modality

### Noise scales
sigma_min = torch.min(sigma_data).item()/10
sigma_max = 0.9
N = 20 # number of noise scales

### Log-normal distribution of noise scales
rho = 4 # from "Elucidating the Design Space of Diffusion-Based Generative Models"
noise_scales = ( sigma_min**(1/rho) + torch.arange(N)*(sigma_max**(1/rho) - sigma_min**(1/rho))/(N-1) )**rho

### Denoising
test_size = 1000 # dimension of data


#%% Neural Networks for the denoisers

### All in one denoiser
class Denoiser(nn.Module):
    def __init__(self, width, data_size):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(nn.Linear(data_size+1, self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.width), nn.ReLU(),
                                 nn.Linear(self.width, data_size) )
        
    def forward(self, x, alpha):
        """Takes noised data and predict noise level."""
        with torch.no_grad():
            s = alpha*torch.ones((x.shape[0], 1))
        return self.net( torch.cat((x,s), dim=1) )

den = Denoiser(width, data_size) 


#%% Training loop with iteration on the denoisers


losses = np.zeros(nb_epochs)
optimizer = torch.optim.SGD(den.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

for epoch in range(nb_epochs):
    
    id_noise_scale = random.randint(0, len(noise_scales)-1)
    alpha = 1 - noise_scales[id_noise_scale]
    alpha_bar = torch.prod(1 - noise_scales[:id_noise_scale+1])
    
    x0 = torch.randn((batch_size, data_size)) * sigma_data[0,:] + mu_data[0,:] # initial unnoised data
    for mod in range(1, multimodality):
        x0 = torch.cat((x0, torch.randn((batch_size, data_size))*sigma_data[mod,:] + mu_data[mod,:]), dim=0)
    
    eps = torch.randn_like(x0) # noise
    x_noised = np.sqrt(alpha_bar)*x0 + np.sqrt(1-alpha_bar)*eps # adding noise corresponding to noise_scale
    pred = den(x_noised, alpha) # prediction of the noise level        
    
    # loss = torch.linalg.vector_norm(eps - pred) # difference between actual noise and prediction
    loss = torch.sum( (eps - pred)**2, dim=(0,1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses[epoch] = loss.detach().item()
   
    
plt.title("Training loss")
plt.plot(np.arange(nb_epochs), losses)
plt.show()
    

#%% Denoising process

def plot_sample(x, id_noise_scale):
    plt.title(f"N = {id_noise_scale}   beta = {noise_scales[id_noise_scale]:.3f}")
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
    

### Sample from the most noised distribution
alpha_bar = torch.prod(1 - noise_scales)
x = torch.randn((test_size, data_size))*torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar)
plot_sample(x, len(noise_scales)-1)
convergence_measure(x)

for id_noise_scale in range(len(noise_scales)-1, -1, -1):
    if id_noise_scale > 1:
        z = torch.randn_like(x)
    else:
        z = torch.zeros_like(x)
        
    alpha = 1 - noise_scales[id_noise_scale]
    alpha_bar = torch.prod(1 - noise_scales[:id_noise_scale+1])
    
    # sigma_sq = noise_scales[id_noise_scale] # Two choices for sigma
    sigma_sq = noise_scales[id_noise_scale] * (1 - alpha_bar/alpha)/(1 - alpha_bar)
    with torch.no_grad():
        x = (x - (1-alpha)*den(x, alpha)/np.sqrt(1-alpha_bar) )/np.sqrt(alpha) + torch.sqrt(sigma_sq)*z
    plot_sample(x, id_noise_scale)
    convergence_measure(x)

### Repeat last step for further denoising
for repeat in range(10):
    with torch.no_grad():
        x = (x - (1-alpha)*den(x, alpha)/np.sqrt(1-alpha_bar) )/np.sqrt(alpha) + torch.sqrt(sigma_sq)*z
    plot_sample(x, id_noise_scale)
    convergence_measure(x)