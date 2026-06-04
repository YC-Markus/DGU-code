import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SpatialFFN(nn.Module):
    def __init__(self, dim: int, expansion_ratio: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * expansion_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * expansion_ratio, dim, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)





class DeformableSelfAttentionOptimized(nn.Module):
    def __init__(
        self,
        dim_global: int = 96,
        num_heads: int = 4,
        num_points: int = 4,
        window_size: int = 3,
        offset_downsample: int = 2, 
    ):
        super().__init__()
        self.dim_global = dim_global
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim_global // num_heads
        self.window_size = window_size
        self.offset_downsample = offset_downsample

        self.query_proj = nn.Conv2d(dim_global, dim_global, kernel_size=1)
        self.kv_proj = nn.Conv2d(dim_global, 2 * dim_global, kernel_size=1)

        self.offset_proj = nn.Sequential(
            nn.Conv2d(dim_global, dim_global, kernel_size=offset_downsample, stride=offset_downsample),
            nn.GELU(),
            nn.Conv2d(dim_global, num_heads * num_points * 2, kernel_size=1)
        )
        
        self.register_buffer("grid", None, persistent=False)

        self.out_proj = nn.Conv2d(dim_global, dim_global, kernel_size=1)
        self.norm = nn.GroupNorm(1, dim_global)

    def _get_grid(self, H, W, device):
        if self.grid is None or self.grid.shape[3:5] != (H, W):
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij'
            )
            self.grid = torch.stack([x, y], dim=-1).view(1, 1, H, W, 2) # (1, 1, H, W, 2)
        return self.grid
    forward_count = 0 



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_norm = self.norm(x)

        off = self.offset_proj(x_norm)
        
        if self.offset_downsample > 1:
            off = F.interpolate(off, size=(H, W), mode='bilinear', align_corners=True)

        off = off.view(B, self.num_heads, self.num_points, 2, H, W)
        off = off.permute(0, 1, 4, 5, 2, 3) # (B, heads, H, W, pts, 2)

        off_scale = torch.tensor([2.0 * self.window_size / W, 2.0 * self.window_size / H], device=x.device)
        off = torch.tanh(off) * off_scale.view(1, 1, 1, 1, 1, 2)

        base_grid = self._get_grid(H, W, x.device) # (1, 1, H, W, 2)
        sample_locs = base_grid.unsqueeze(4) + off # (B, heads, H, W, pts, 2)
        
      
        sample_locs = sample_locs.permute(0, 1, 4, 2, 3, 5).reshape(B * self.num_heads, self.num_points, H, W, 2)
        sample_locs = sample_locs.permute(0, 2, 3, 1, 4).reshape(B * self.num_heads, H, W * self.num_points, 2)

        kv = self.kv_proj(x).view(B * self.num_heads, 2 * self.head_dim, H, W)
        
        sampled_kv = F.grid_sample(
            kv, sample_locs, mode='bilinear', padding_mode='zeros', align_corners=True
        )

        sampled_kv = sampled_kv.view(B, self.num_heads, 2, self.head_dim, H, W, self.num_points)
        k, v = sampled_kv.unbind(2)

        q = self.query_proj(x_norm).view(B, self.num_heads, self.head_dim, H, W, 1)

        attn_weights = (q * k).sum(dim=2) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = (v * attn_weights.unsqueeze(2)).sum(dim=-1)
        out = out.reshape(B, C, H, W)

        return x + self.out_proj(out)