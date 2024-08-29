import torch
from torch import nn
from model import DiT

class DiTConfig:
    """
    a simple small config
    """
    def __init__(self):
        self.img_size = 256  # Size of the input images
        self.patch_size = 4   # Size of each patch
        self.dim = 768        # Embedding dimension
        self.depth = 12       # Number of transformer blocks
        self.heads = 12       # Number of attention heads
        self.mlp_dim = 3072   # Dimensions of the multilayer perceptron in the transformer block
        self.in_channels = 3  # Number of input channels
        self.timesteps = 1000 # Number of timesteps for diffusion
        self.beta_start = 0.0001
        self.beta_end = 0.02

    def start(self):
        return DiT(self)
