import torch
import torch.nn as nn
import numpy as np

class LayerNorm(nn.Module):
    """
     simple layer norm 
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps)

class SelfAttention(nn.Module):
    """
    simple multi-head self-attention layer
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.to_qkv(x).reshape(B, L, 3, self.heads, D // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.to_out(x)

class TransformerBlock(nn.Module):
    """
    transformer block with self-attention and multi-layer perceptron
    """
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.ln2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attn(x)
        x = self.ln2(x)
        x = x + self.mlp(x)
        return x
