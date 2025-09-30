# Task 1: Smaller components - Patchify, TimeEmbedding, AdaLN-Zero
import torch
import torch.nn as nn
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    def __init__(self, output_dim: int):        
        super().__init__() 
        self.output_dim = output_dim
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, output_dim, 2).float() / output_dim))
        self.register_buffer('inv_freq', inv_freq) 
        
        hidden_dim = output_dim * 4 
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float() # shape (B,) 
        
        # for faster computation of all batches simulataneously
        sinusoidal_input = torch.einsum("i,j->ij", timesteps, self.inv_freq)
        pos_encoding = torch.cat([sinusoidal_input.sin(), sinusoidal_input.cos()], dim=-1) # shape (B, output_dim)            
        time_cond = self.mlp(pos_encoding)
        
        return time_cond
        

class Patchify(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
                
        # (B, EmbedDim, H, W) -> (B, EmbedDim, H*W) -> (B, H*W, EmbedDim)
        # where H*W is the sequence length
        # output shape:(B, SequenceLength, EmbedDim)
        
        x = x.flatten(2).transpose(1, 2)
        
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        return self.proj(attn_output) 


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, time_embed_dim: int):
        super().__init__()
                
        self.linear_time_proj = nn.Linear(time_embed_dim, 6 * embed_dim)
        
        self.attn = Attention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, c: torch.Tensor):

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = \
            self.linear_time_proj(c).chunk(6, dim=1) 
        
        shape = (x.size(0), 1, x.size(-1))
        
        gamma1, beta1, alpha1 = gamma1.view(shape), beta1.view(shape), alpha1.view(shape)
        gamma2, beta2, alpha2 = gamma2.view(shape), beta2.view(shape), alpha2.view(shape)

        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + gamma1) + beta1 
        
        attn_output = self.attn(norm_x)        
        x = x + alpha1 * attn_output         
        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + gamma2) + beta2
        
        ffn_output = self.ffn(norm_x)        
        x = x + alpha2 * ffn_output 

        return x