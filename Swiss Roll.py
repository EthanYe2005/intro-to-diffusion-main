import torch
import random
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)
print(f"Using device: {device}")
random.seed(2778)
np.random.seed(2778)

# 准备工作
diffusion_steps = 1000
betas = torch.linspace(1e-4, 0.02, diffusion_steps).to(device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0).to(device)

# 时间编码器
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: [batch_size, 1]
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings[None, :] # [batch, 1] * [1, half_dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings # [batch, dim]
    
# 预测噪声器，输出的是预测的噪声
class Denoise(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.time_dim = 32
        self.time_mlp = SinusoidalPositionEmbeddings(self.time_dim)
        self.net = nn.Sequential(nn.Linear(input_size+self.time_dim, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, input_size))
        
    def forward(self, x, phase):
        embedding_t = self.time_mlp(phase)
        x = torch.cat((x, embedding_t), dim=1)
        return self.net(x)

#数据生成函数
def gen_swiss_roll(batch_size):
    theta = torch.rand((batch_size, 1))*4*math.pi
    data_x = theta * torch.cos(theta)/6.0
    data_y = theta * torch.sin(theta)/6.0
    ans = torch.cat((data_x, data_y), dim=1).to(device)
    return ans

# training loop
Denoiser = Denoise(2, 256).to(device)
optimizer = torch.optim.Adam(params=Denoiser.parameters())

def train(epoch):
    losses = []
    for i in range(epoch):
        train_data = gen_swiss_roll(128)
        epsilon = torch.randn_like(train_data)
        phase = torch.randint(0, diffusion_steps, (train_data.shape[0],1),device=device)

        pred_epsilon = Denoiser(torch.sqrt(alpha_bar[phase])*train_data + torch.sqrt(1-alpha_bar[phase])*epsilon, phase)

        loss = torch.mean((epsilon-pred_epsilon)**2)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses

# test 
def test():
    print("Generating samples...")
    data = torch.randn((1000,2),device=device)
    for j in reversed(range(diffusion_steps)):
        phase = torch.full((data.shape[0], 1), j, device=device, dtype=torch.float32)
        pred_epsilon = Denoiser(data, phase)
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
    return data

if __name__== '__main__':
    # 1. 训练
    losses = train(epoch= 20000)
    print("Training is over.")
    # 2. 生成测试数据
    generated_data = test()
    generated_data = generated_data.detach().cpu().numpy()

    # 3. 准备一些真实的训练数据用于对比
    with torch.no_grad():
        real_data = gen_swiss_roll(1000).cpu().numpy()
        

    # 4. 开始绘图
    plt.figure(figsize=(12, 5))

    # 子图 1：真实的训练数据
    plt.subplot(1, 2, 1)
    plt.scatter(real_data[:, 0], real_data[:, 1], s=5, alpha=0.5, color='blue')
    plt.title("Ground Truth (Swiss Roll)")
    plt.grid(True)
    plt.xlim(-3, 3) # 根据你的归一化结果调整范围
    plt.ylim(-3, 3)

    # 子图 2：生成的测试数据
    plt.subplot(1, 2, 2)
    plt.scatter(generated_data[:, 0], generated_data[:, 1], s=5, alpha=0.5, color='green')
    plt.title("Generated Data (Diffusion Model)")
    plt.grid(True)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.tight_layout()
    plt.show()
