# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:03:46 2023
@author: Jean-Baptiste Bouvier

Create a video of multimodal diffusion in 2D.
Network is trained to remove one noise scale at a time.
Makes a video and gif for the denoising of the 4 Gaussians
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
frame_per_second = 24
title = "denoising_4"

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
N = 17 # number of noise scales

### Log-normal distribution of noise scales
rho = 2 # from "Elucidating the Design Space of Diffusion-Based Generative Models"
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
X = torch.zeros((N+1, test_size, data_size))
X[0] = torch.randn((test_size, data_size))*sigma_max # sample from the most noised distribution


for i in range(N):
    with torch.no_grad():
        X[i+1] = X[i] - denoiser(X[i], Sigmas[i]) # remove one level of noise

#%% Interpolation
num_frames = duration*frame_per_second
Y = torch.zeros((num_frames, test_size, data_size))
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
    ax.scatter(Y[i,:,0], Y[i,:,1], s=10)
    ax.set_xlim([-3*sigma_max, 3*sigma_max])
    ax.set_ylim([-3*sigma_max, 3*sigma_max])
    ax.axis("off")
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration=duration)
animation.write_videofile(title + ".mp4", fps=frame_per_second)


fig, ax = plt.subplots()
def make_frame_out_and_back(t):
    
    i = round(t * frame_per_second)
    if t >= duration:
        i -= 2*(i - num_frames+1)
    ax.clear()
    ax.scatter(Y[i,:,0], Y[i,:,1], s=10)
    ax.set_xlim([-3*sigma_max, 3*sigma_max])
    ax.set_ylim([-3*sigma_max, 3*sigma_max])
    ax.axis("off")
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame_out_and_back, duration=2*duration)
animation.write_gif(title + ".gif", fps=frame_per_second)