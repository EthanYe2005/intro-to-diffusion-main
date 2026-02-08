import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

# 准备工作：数据服从N(1,1)
data_mean = 1.0
data_var = 1.0

diffusion_steps = 1000
betas = torch.linspace(1e-4, 0.02, diffusion_steps)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

# 预测噪声器，输出的是预测的噪声
class Denoise(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size+1, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, 1))
        
    def forward(self, x, phase):
        x = torch.cat((x, phase*torch.ones_like(x)/diffusion_steps), dim=1)
        return self.net(x)

# training loop
Denoiser = Denoise(1, 32)
optimizer = torch.optim.Adam(params=Denoiser.parameters())

def train(x, epoch):
    losses = []
    for i in range(epoch):
        epsilon = torch.randn_like(x)
        phase = torch.randint(0, diffusion_steps, x.size())

        pred_epsilon = Denoiser(torch.sqrt(alpha_bar[phase])*x + torch.sqrt(1-alpha_bar[phase])*epsilon, phase)

        loss = torch.mean((epsilon-pred_epsilon)**2)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses

# test 
def test():
    all_data = []
    print("Generating samples...")
    for i in range(200):
        data = torch.randn((32,1))
        for j in reversed(range(diffusion_steps)):
            pred_epsilon = Denoiser(data, j)
            alpha_j = alphas[j]         # 当前步的 alpha
            alpha_bar_j = alpha_bar[j]  # 当前步的 alpha_bar
            
            coef1 = 1 / torch.sqrt(alpha_j)
            coef2 = (1 - alpha_j) / torch.sqrt(1 - alpha_bar_j)
            
            mean = coef1 * (data - coef2 * pred_epsilon)
            
            if j > 0:
                noise = torch.randn_like(data)
                sigma_j = torch.sqrt(betas[j]) 
                data = mean + sigma_j * noise
            else:
                data = mean
        
        all_data.append(data.detach().cpu())
        
        mean = torch.mean(data).item()
        var = torch.var(data).item()
        if(i%20 == 0):
            print("Batch {}: Current mean = {:.4f}, variance = {:.4f}".format(i, mean, var))
    
    return torch.cat(all_data, dim=0).numpy().flatten()
        

if __name__== '__main__':
    train_data = torch.randn((32,1))+1

    print('Start training.')
    losses = train(train_data, epoch=50000)

    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

    print('Start testing.')
    generated_data = test()

    plt.figure()
    plt.hist(generated_data, bins=50, density=True, alpha=0.6, color='g', label='Generated Data')
    x = np.linspace(-3, 5, 100)
    p = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*((x-1)**2))
    plt.plot(x, p, 'r', linewidth=2, label='N(1,1)')
    plt.legend()
    plt.title('Data Distribution')
    plt.show()


