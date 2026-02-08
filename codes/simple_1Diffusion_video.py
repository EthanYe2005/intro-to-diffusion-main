# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:03:00 2023
@author: Jean-Baptiste Bouvier

Simple implementation of 1D Diffusion Model
trained to remove one noise scale at a time.
Makes a video of the random points being denoised
"""

import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#%% Hyperparameters

### Video
duration = 2 # [seconds]
frame_per_second = 20
title = "denoising_1"

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
Sigmas = torch.cat((Sigmas, torch.tensor([0.])), dim=0)

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

X = torch.zeros((N+1, test_size, 1))
X[0] = torch.randn((test_size, 1))*sigma_max # sample from the most noised distribution

for i in range(N):
    with torch.no_grad():
        X[i+1] = X[i] - denoiser(X[i], Sigmas[i]) # remove one level of noise

#%% Interpolation to get intermediate frames

num_frames = duration*frame_per_second
Y = torch.zeros((num_frames, test_size, 1))
Y[0] = X[0]
for frame_id in range(num_frames):
    coef = frame_id*N/num_frames
    i = round(np.floor(coef))
    alpha = coef - i
    Y[frame_id] = X[i]*(1-alpha) + alpha*X[i+1]


#%% Video making

fig, ax = plt.subplots()
def make_frame(t):
    i = round(t * frame_per_second)
    ax.clear()
    ax.scatter(np.arange(test_size), Y[i,:,0].numpy(), s=10)
    ax.set_ylim([-3, 3])
    ax.axis("off")
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration=duration)
animation.write_videofile(title + ".mp4", fps=frame_per_second)

