import torch
from torch import nn
import numpy as np

from utils import (init_linear_module, init_layernorm_module,
                   init_clstoken, init_posembed)


vit_small = lambda drop_path: ViT(embed_dim=384, num_heads=6, num_blks=12, 
                                  patch_size=8, drop_path=drop_path)


class MSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.linear1 = nn.Linear(embed_dim, embed_dim * 3)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

        # 参数初始化
        init_linear_module(self.linear1)
        init_linear_module(self.linear2)
        
    def forward(self, x):
        m, n, d = x.shape
        qkv = self.linear1(x).reshape(m,n,3,self.num_heads,-1)
        qkv = qkv.permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (self.embed_dim / self.num_heads) ** -0.5
        weights = (q @ k.transpose(-2,-1)) * scale
        weights = weights.softmax(dim=-1)

        attn = (weights @ v).transpose(1,2).reshape(m,n,d)
        attn = self.linear2(attn)

        return attn
    

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

        # 参数初始化
        init_linear_module(self.linear1)
        init_linear_module(self.linear2)

    def forward(self, x):
        return self.linear2(self.act1(self.linear1(x)))
    

class DropBatch(nn.Module):
    def __init__(self, drop_path_rate=0):
        super().__init__()
        self.rate = drop_path_rate

    def forward(self, x):
        if self.rate == 0:
            return x
        mask = torch.rand(x.shape[0], device=x.device) > self.rate
        mask = mask[:,None,None]
        return mask * x


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_path=0):
        super().__init__()
        self.mlp = MLP(embed_dim, embed_dim * 4)
        self.msa = MSA(embed_dim, num_heads)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.drop = DropBatch(drop_path) if drop_path > 0 else nn.Identity()

        # 参数初始化
        init_layernorm_module(self.ln1)
        init_layernorm_module(self.ln2)

    def forward(self, x):
        x = x + self.drop(self.msa(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x
    

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3, out_channels=embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1,2)


class ClassEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))

        # 参数初始化
        init_clstoken(self.cls_token)

    def forward(self, x):
        m = x.shape[0]
        return torch.cat((self.cls_token.expand(m,-1,-1), x), dim=1)


class PosEmbed(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1,785,embed_dim))

        # 参数初始化
        init_posembed(self.pos_embed)

    def interpolate(self, image_width, image_height):
        width = image_width // self.patch_size
        height = image_height // self.patch_size
        pos_embed = nn.functional.interpolate(
            input=self.pos_embed[:,1:,:].transpose(1,2).reshape(1,self.embed_dim,28,28),
            size=(width, height), mode='bicubic'
        ).flatten(2).transpose(1,2)
        return torch.cat((self.pos_embed[:,0,:][:,None,:], pos_embed), dim=1)
    
    def forward(self, x, image_height, image_width):
        return x + self.interpolate(image_width, image_height)


class ViT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blks, patch_size, drop_path):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blks = num_blks
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(embed_dim, patch_size)
        self.class_embed = ClassEmbed(embed_dim)
        self.pos_embed = PosEmbed(embed_dim, patch_size)
        

        drop_path = np.linspace(0, drop_path, num_blks)
        seqs = [ViTBlock(embed_dim, num_heads, drop_path[i]) for i in range(num_blks)]
        self.blks = nn.ModuleList(seqs)

        self.ln = nn.LayerNorm(embed_dim)

        # 参数初始化
        init_layernorm_module(self.ln)

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        x = self.class_embed(x)
        x = self.pos_embed(x, H, W)

        for blk in self.blks:
            x = blk(x)

        return self.ln(x)[:,0]
