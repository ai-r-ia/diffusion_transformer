# Task 2: Main DiT model implementation
import torch
import torch.nn as nn
from utils.modules import TimeEmbedding, Patchify, DiTBlock 

class DiT(nn.Module):
    def __init__(self, 
                img_size: int, 
                in_channels: int, 
                patch_size: int, 
                embed_dim: int, 
                num_heads: int, 
                depth: int,
                num_classes: int):
        
        super().__init__()
        
        self.img_size = img_size 
        self.in_channels = in_channels
        self.out_channels = in_channels 
        self.patch_size = patch_size
        
        
        self.time_embed = TimeEmbedding(embed_dim)
        self.class_embed = nn.Embedding(num_classes, embed_dim) 

        self.patchify = Patchify(in_channels, embed_dim, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, embed_dim) 
            for _ in range(depth)
        ])


        self.fc = nn.Linear(embed_dim, patch_size**2 * self.out_channels)
        self.unpatchify_size = (img_size // patch_size)
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        # x shape: (B, C, H, W)
        
        t_embed = self.time_embed(t) 
        y_embed = self.class_embed(y)        
        c = t_embed + y_embed # (B, EmbedDim)

        x = self.patchify(x) # (B, SeqLen, EmbedDim)
    
        for block in self.blocks:
            x = block(x, c) 
            
        x = self.fc(x) # (B, SeqLen, PatchSize^2 * OutChannels)
        
        B, S, V = x.shape
        x = x.transpose(1, 2).contiguous().view(B, self.out_channels, self.unpatchify_size, self.unpatchify_size, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(B, self.out_channels, self.img_size, self.img_size)
        pred_noise = x 
        
        return pred_noise