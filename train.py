import torch
import torch.nn as nn
import torch.optim as optim
from model import DiT
from config import DiTConfig

def train_model():
    # configuration
    config = DiTConfig()
    model = DiT(config)
    
    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # training loop
    for epoch in range(5):
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # random input and target
        x = torch.randn(8, 3, config.img_size, config.img_size)
        t = torch.randint(0, config.timesteps, (8,))
        
        # forward pass where we get the noisy image and the noise
        xt, noise = model.diffusion.diffuse(x, t)
        output = model(xt, t)
        
        # loss calculation to get the loss between the noise and the output
        loss = criterion(output, noise)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == "__main__":
    train_model()
