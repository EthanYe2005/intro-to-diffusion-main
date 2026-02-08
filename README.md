# Introduction to Diffusion models

This repository provides an introduction to Diffusion models with simple examples coded in Pytorch.
There are already many good resources explaining how diffusion models work, such as [Lilian Weng's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) or even [Wikipedia](https://en.wikipedia.org/wiki/Diffusion_model).
However, having a basic understanding of how diffusion works is not enough to code them.
Most available codes for diffusion models are aimed at generating images and hence are very complex.
Here we will instead focus on simplicity and implement a simple diffusion model in just 100 lines of python code!


## What are diffusion models?

Diffusion models belong to the class of generative machine learning as they can create new content and are very performant at image generation.
In short, diffusion models learn to iteratively remove some noise from the data.
Starting from a purely stochastic signal, a white noise, a trained diffusion model can iteratively decrease the noise level of this data until converging to a given distribution.
Here we will focus on extremely simple models, and train a neural network to remove noise from 1D data as illustrated below:

![noise scales](assets/noise_scales.png)


## How to implement a diffusion model in 100 lines of python?

In our first implementation, the only libraries needed are [torch](https://pytorch.org/) for the neural networks, [matplotlib](https://matplotlib.org/) for plotting our results, and [numpy](https://numpy.org/) for the calculations.
The simplest code is `simple_1Diffusion.py` in the `codes` folder which implements a basic diffusion model for 1D data and is self-contained in 100 lines of python!

First, we define the data distribution as a Gaussian of mean `mu_data = 1` and standard deviation (std) `sigma_data = 0.01`.
The goal will be to recover this data distribution from a unit Gaussian noise of mean 0 and std 1 as illustrated below.

![Transformation of a Gaussian noise of mean 0 and std 1 into the data distribution of mean 1 and std 0.01](assets/pdf_diffusion_gif.gif)

The denoiser neural network (dnn) will be trained with a learning rate `lr = 1e-3`, a `batch_size = 32` and iterated for `nb_epochs = 1000`.
We define the minimal and maximal std of the noise levels of the dnn `sigma_min` and `sigma_max`.
These noise levels are spread following a log-normal scale as suggested by the paper ["Elucidating the Design Space of Diffusion-Based Generative Models"](https://arxiv.org/abs/2206.00364).

The dnn is coded as a `class Denoiser` with a neural network taking as inputs the data `x` and its noise level `sigma` before returning its prediction for the noise added to some clean data.
We use a simple Multi-Layer Perceptron with ReLU activations as our neural network.

The training loop randomly chooses a noise level `sigma` and creates a noise signal `n` to be added to the clean data `y` parametrized by `mu_data` and `sigma_data`.
The dnn takes as inputs `y+n` and `sigma`, and is trained to predict the extra noise corresponding to a transition between levels `sigma` and the next noise level.
We calculate a quadratic loss and use stochastic gradient descent (SGD) to optimize the dnn.
   
The denoising process starts from a stochastic signal of mean 0 and std 1.
From this signal we iteratively remove one noise level using the trained dnn until obtaining a cleaned signal corresponding to the initial data distribution of mean `mu_data` and std `sigma_data`.
We can even make a video of this quick denoising process with `simple_1Diffusion_video.py`.


## Diffusion models in 2D

Now that we have a 1D denoiser, we can scale it up to 2D with minimal changes and obtain the code `simple_2Diffusion.py`.



## Diffusion models in 2D with multimodalities

Diffusion models are supposed to be very expressive, allowing them to capture multimodalities present in the initial data.
We will illustrate this with an initial data distribution being the sum of 4 narrow Gaussians with spikes at (-1, -1), (-1, 1), (1, -1) and (1, 1).
As before, we will add noise until this initial distribution is indistinguishable from a Gaussian centered at the origin with std 1.
The denoising process should then separate the data into the four spikes as illustrated on the gif below.

![Noising and denoising a multimodal distribution](assets/denoising_4.gif)

The code to implement this 2D multimodal diffusion process is `simple_multimodal.py` and the gif can be generated with `simple_multimodal_video.py`.


## Denoising Diffusion Probabilistic Models (DDPM)

The diffusion models presented so far use a naive denoising process which works on our simple cases.
To obtain better quality diffusion models we will follow the implementation of ["Denoising Diffusion Probabilistic Models"](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) (DDPM).
The simplest code for 1D data is `single_DDPM.py` which can also be found in the `codes` folder.
Note that `single_DDPM.py` uses a single neural network to denoise each noise level, but we could also have one neural network for each noise level as implemented in `simple_DDPM.py`.
As we did previously, we can also extend DDPM to 2D multimodal data distributions with the code `multimodal_DDPM.py`.
Finally, to get a better understanding of the evolution of the probability density functions (pdf) through the different noise scales you can look at `detailed_DDPM.py`.



