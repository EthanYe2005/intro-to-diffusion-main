# -*- coding: utf-8 -*-
"""
Created on Wed Oct 4 09:17:54 2023

@author: Jean-Baptiste Bouvier

Detailed implementation of Denoising Diffusion Probabilistic Models (DDPM) 
Paper: https://arxiv.org/abs/2006.11239
Notations from:
SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS
and Denoising Diffusion Probabilistic Models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



#%% Functions for the pdf

def pdf(x, mu, sigma):
    return torch.exp(-0.5*((x-mu)/sigma)**2 )/(sigma*np.sqrt(2*np.pi))

def pdf_data(x):
    return pdf(x, mu_data, sigma_data)

def plot_pdf(mu, sigma, x_min, x_max):
    step=(x_max-x_min)/1000
    X = torch.arange(start=x_min, end=x_max, step=step)
    Y = pdf(X, mu, sigma)
    plt.title(f'pdf of mean {mu:.2f} and standard deviation {sigma:.2f}')
    plt.scatter(X, Y, s=10)
    plt.show()
    print(f"Integral of plotted pdf = {Y.sum()*step:.3f}")


### P(x_i | x_{i-1})
def single_step_perturbation_kernel(x_noised, x, beta):
    """P(x_noised | x) = N(x_noised; sqrt(1-beta)*x, beta)"""
    grid_x_noised, grid_x = torch.meshgrid((x_noised, x), indexing='ij')
    return pdf(grid_x_noised, torch.sqrt(1-beta)*grid_x, beta)




class perturbed_distribution():
    def __init__(self, id_noise_scale):
        """A class for all the 1D perturbed pdf generated empirically.
        Can be sampled."""
        
        self.id_noise_scale = id_noise_scale
        self.alpha = 1 - noise_scales[id_noise_scale]
        self.alpha_bar = torch.prod(1 - noise_scales[:id_noise_scale+1])

        self.x_min = np.sqrt(self.alpha_bar) - 6*np.sqrt(1 - self.alpha_bar)
        self.x_max = np.sqrt(self.alpha_bar) + 6*np.sqrt(1 - self.alpha_bar)
        self.step = (self.x_max - self.x_min)/1000
        ### Discrete support of the distribution
        self.support = torch.arange(start=self.x_min, end=self.x_max, step=self.step)
            
        ### Probability at each point of the discrete support
        self.P = self.multi_step_perturbed_pdf(self.support)*self.step
        self.total_probability = self.P.sum().item()
        self.mean = torch.sum(self.support * self.P).item()
        self.variance = torch.sum( (self.support - self.mean)**2 * self.P ).item()
        self.sigma = np.sqrt(self.variance)
        ### Check
        # print( np.abs(np.sqrt(self.alpha_bar) - self.mean)  )
        # print( np.abs(np.sqrt(1-self.alpha_bar) - self.sigma) )
        
        ### Cumulative distribution function
        self.cdf = torch.cumsum(self.P, dim=0)
        
    def plot_info(self):
        print(f"Integral = {self.total_probability:.3f}  mean = {self.mean:.3f}  sigma = {self.sigma:.3f}")
       
    def plot_pdf(self):
        plt.title(f'Perturbed pdf of noise scale {self.id_noise_scale}')
        plt.scatter(self.support, self.P/self.step, s=10)
        plt.show()
        
    def plot_cdf(self):
        plt.title(f'Perturbed cdf of noise scale {self.id_noise_scale}')
        plt.scatter(self.support, self.cdf, s=10)
        plt.show()
        
    def multi_step_perturbation_kernel(self, x_i, x0):
        """Calculates pdf of x_i | x_0, with i = id_noise_scale"""
        grid_x_i, grid_x0 = torch.meshgrid((x_i, x0), indexing='ij')
        return pdf(grid_x_i, np.sqrt(self.alpha_bar)*grid_x0, np.sqrt(1-self.alpha_bar))

    def multi_step_perturbed_pdf(self, x_tilde):
        """Calculates the probability of x_tilde"""
        x_min = mu_data - 6*sigma_data
        x_max = mu_data + 6*sigma_data
        # No point in integrating past these bounds, pdf_data will be 0
        step = (x_max - x_min)/1000
        X = torch.arange(start=x_min, end=x_max, step=step)
        return torch.sum(pdf_data(X)*self.multi_step_perturbation_kernel(x_tilde, X), dim=1)*step

    def sample(self, N):
        """Samples N points from the distribution"""
        r = torch.rand(N)
        x = torch.zeros((N,1))
        for i in range(N):
            idx = (self.cdf >= r[i]).nonzero(as_tuple = True)[0][0].item()
            x[i,0] = self.support[idx]
        return x
        
        
 
class diffusion_model():
    def __init__(self, noise_scales):
        """Class to gather all the noising distributions of the diffusion process."""
        self.noise_scales = noise_scales
        for id_noise_scale in range(len(noise_scales)):
            setattr(self, f"d_{id_noise_scale}", perturbed_distribution(id_noise_scale) )
            
    def plot_pdf(self, id_noise_scale):
        getattr(self, f"d_{id_noise_scale}").plot_pdf()
    
    def plot_all_pdf(self):
        for id_noise_scale in range(len(self.noise_scales)):
            getattr(self, f"d_{id_noise_scale}").plot_pdf()
            
    def plot_all_info(self):
        for id_noise_scale in range(len(self.noise_scales)):
            getattr(self, f"d_{id_noise_scale}").plot_info() 


#%% Setting up perturbed pdfs
### Noise scales
noise_scales = torch.tensor([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
### Initial data distribution
mu_data = 1
sigma_data = 0.01


### Setup all the noised distributions of the diffusion process
model = diffusion_model(noise_scales)
model.plot_all_info()

### Plot the pdf of the original data and of the perturbed data
plot_pdf(mu_data, sigma_data, mu_data-6*sigma_data, mu_data+6*sigma_data)
model.plot_all_pdf()


#%% Neural Nets for the denoisers
import torch.nn as nn
### Class for single denoiser for a given noise_scale
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
    
### Class to gather all the denoisers, 1 for each step on the scale
class Denoisers():
    def __init__(self, width, noise_scales):
        self.noise_scales = noise_scales
        for id_noise_scale in range(len(noise_scales)):
            setattr(self, f"d_{id_noise_scale}", denoiser(width) )


denoisers = Denoisers(32, noise_scales) 



#%% Training loop
dim = 1000 # dimension of data

### Train each denoiser
for id_noise_scale in range(len(noise_scales)):
    
    # Denoiser for this noise scale           
    den = getattr(denoisers, f"d_{id_noise_scale}")
    
    if id_noise_scale == 0: 
        lr = 3e-3
        nb_epochs = 25000 # first NN needs more training
    else:
        lr = 1e-5
        nb_epochs = 2000
        # Warmstart weights to those of previous noise scale
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
        # optimizer = torch.optim.SGD(den.parameters(), lr/(1+epoch)**0.5, weight_decay=1e-5)
        if loss.item() < 5: lr = 1e-6
        
    plt.title(f"Loss for noise scale {id_noise_scale}")
    plt.plot(np.arange(nb_epochs), losses)
    plt.show()
    

#%% Denoising process

### Once they are trained we can sample from the most noised distribution
id_noise_scale = len(noise_scales)-1
x_noised = getattr(model, f"d_{id_noise_scale}").sample(dim)

### Plot noised sample
plt.title("Noised sample")
plt.scatter(np.arange(dim), x_noised.numpy())
plt.show()
print(f"Noised sample:   mean {x_noised.mean().item():.3f}  std {x_noised.std().item():.3f}")

for id_noise_scale in range(len(noise_scales)-1, 0, -1):
    if id_noise_scale > 1:
        z = torch.randn_like(x_noised)
    else:
        z = torch.zeros_like(x_noised)
        
    alpha = 1 - noise_scales[id_noise_scale]
    alpha_bar = torch.prod(1 - noise_scales[:id_noise_scale+1])
    den = getattr(denoisers, f"d_{id_noise_scale}")
    # sigma_sq = noise_scales[id_noise_scale]
    sigma_sq = noise_scales[id_noise_scale] * (1 - alpha_bar/alpha)/(1 - alpha_bar)
    x_noised = (x_noised - (1-alpha)*den(x_noised)/np.sqrt(1-alpha_bar) )/np.sqrt(alpha) + torch.sqrt(sigma_sq)*z

plt.title("Denoised sample")
plt.scatter(np.arange(dim), x_noised.detach().numpy())
plt.show()

print(f"Denoised sample: mean {x_noised.mean().item():.3f}  std {x_noised.std().item():.3f}")
print(f"Denoising goal:  mean {model.d_0.mean:.3f}  std {model.d_0.sigma:.3f}")

