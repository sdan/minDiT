import torch
import torch.nn as nn
from transformer import TransformerBlock
from diffusion import Diffusion

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion = Diffusion(config)

        # patchify turns a 2D image into a 1D sequence of tokens see fig 4
        self.patchify = nn.Conv2d(config.in_channels, config.dim, 
                                  kernel_size=config.patch_size, 
                                  stride=config.patch_size)

        # ViT frequency-based positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, config.img_size // config.patch_size, config.dim))

        # transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config.dim, config.heads, config.mlp_dim) for _ in range(config.depth)]
        )

        # decoder takes the sequence of tokens and turns it back into a 2D image
        self.decoder = nn.Linear(config.dim, config.patch_size * config.patch_size * config.in_channels)

    def forward(self, x, t):
        """
        Forward pass through the DiT model.
        """
        # patchify turns a 2D image into a 1D sequence of tokens see fig 4
        x = self.patchify(x).flatten(2).transpose(1, 2)
        
        # add positional embeddings
        x = x + self.pos_embedding
        
        # pass through transformer blocks
        x = self.transformer_blocks(x)
        
        # decoder takes the sequence of tokens and turns it back into a 2D image
        x = self.decoder(x).transpose(1, 2).contiguous().view(x.size(0), self.config.in_channels, self.config.img_size, self.config.img_size)
        
        return x

    def sample(self, num_samples, steps, seed=None):
        """
        samples new images from the model using reverse diffusio
        """

        # start from pure noise
        x = torch.randn(num_samples, self.config.in_channels, self.config.img_size, self.config.img_size).to(self.patchify.weight.device)

        # reverse diffusion process
        for t in reversed(range(steps)):
            t_tensor = torch.tensor([t] * num_samples, dtype=torch.long).to(x.device)
            noise_pred = self(x, t_tensor)
            x = self.diffusion.reverse_diffusion_step(x, noise_pred, t)

        return x