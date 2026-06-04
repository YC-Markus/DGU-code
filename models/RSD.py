import torch
import torch.nn as nn
import torch.nn.functional as F





class PixelUnshuffle1D(nn.Module):
    """
    Input: [B, C, N, W] -> Output: [B, C*r, N, W/r]
    """
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        b, c, n, w = x.shape
        r = self.downscale_factor
        if w % r != 0:
            raise ValueError(f"Width {w} is not divisible by downscale factor {r}")
        
        x = x.view(b, c, n, w // r, r)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(b, c * r, n, w // r)
        return x
    

class StripConvResBlock(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=groups),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.net(x)



class SinogramDecomposition(nn.Module):
    def __init__(self, resolutions, perc_channels=[16,24,32,48], H=32):
        super().__init__()
        self.resolutions = resolutions
        self.phys_pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        self.perc_conv256 = nn.Sequential(
            nn.Conv2d(1, perc_channels[0], 3, 1, 1), 
            nn.GELU(),
            StripConvResBlock(perc_channels[0], groups=4),
            nn.Conv2d(perc_channels[0], perc_channels[0], kernel_size=3, padding=1),
        )
        
      
        self.perc_unshuffle128 = PixelUnshuffle1D(2)
        self.perc_conv128 = nn.Sequential(
            nn.Conv2d(perc_channels[0]*2, perc_channels[1], 1),
            nn.GELU(), 
            StripConvResBlock(perc_channels[1], groups=4),
            nn.Conv2d(perc_channels[1], perc_channels[1], kernel_size=3, padding=1),
        )
        
    
        self.perc_unshuffle64 = PixelUnshuffle1D(2)
        self.perc_conv64 = nn.Sequential(
            nn.Conv2d(perc_channels[1]*2, perc_channels[2], 1), 
            nn.GELU(), 
            nn.Conv2d(perc_channels[2], perc_channels[2], kernel_size=3, padding=1),
            SparseSinogramFusionAttention(dim=perc_channels[2], H=H, W=resolutions[1], num_heads=1)
        )
        
      
        self.perc_unshuffle32 = PixelUnshuffle1D(2)
        self.perc_conv32 = nn.Sequential(
            nn.Conv2d(perc_channels[2]*2, perc_channels[3], 1), 
            nn.GELU(), 
            nn.Conv2d(perc_channels[3], perc_channels[3], kernel_size=3, padding=1),
            SparseSinogramFusionAttention(dim=perc_channels[3], H=H, W=resolutions[0], num_heads=1)
        )


        self.perc_unshuffle16 = PixelUnshuffle1D(2)
        self.perc_conv16 = nn.Sequential(
            nn.Conv2d(perc_channels[3]*2, perc_channels[4], 1), 
            nn.GELU(),
            nn.Conv2d(perc_channels[4], perc_channels[4], kernel_size=3, padding=1)
        )




    def forward(self, x):
        p256 = x
        p128 = self.phys_pool(p256)
        p64 = self.phys_pool(p128)
        p32 = self.phys_pool(p64)
        p16 = self.phys_pool(p32)
        phys_outputs = {self.resolutions[-1]: p256, self.resolutions[-2]: p128/2, self.resolutions[-3]: p64/4, self.resolutions[-4]: p32/8, self.resolutions[0]//2: p16/16}
        
        norm_factor = 100
        feat256 = self.perc_conv256(x/norm_factor)
        feat128 = self.perc_conv128(self.perc_unshuffle128(feat256))
        feat64 = self.perc_conv64(self.perc_unshuffle64(feat128))
        feat32 = self.perc_conv32(self.perc_unshuffle32(feat64))
        feat16 = self.perc_conv16(self.perc_unshuffle16(feat32))
        
        perc_outputs = {self.resolutions[0]//2: feat16*norm_factor/16, self.resolutions[-4]: feat32*norm_factor/8, self.resolutions[-3]: feat64*norm_factor/4, 
                        self.resolutions[-2]: feat128*norm_factor/2, self.resolutions[-1]: feat256*norm_factor} 
        return phys_outputs, perc_outputs









def precompute_rope_freqs(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    
    freqs = torch.outer(t, freqs)
    
    freqs_cos = torch.cos(freqs)  # [seq_len, dim/2]
    freqs_sin = torch.sin(freqs)  # [seq_len, dim/2]
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
    
    return freqs_cos, freqs_sin


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    head_dim = x.shape[-1]
    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * freqs_cos) + (x_rotated * freqs_sin)




class AxialAttention1D_RoPE(nn.Module):
    def __init__(self, dim, num_heads, seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        freqs_cos, freqs_sin = precompute_rope_freqs(self.head_dim, seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, x):
        N, L, C = x.shape
        
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        freqs_cos = self.freqs_cos.view(1, 1, L, self.head_dim)
        freqs_sin = self.freqs_sin.view(1, 1, L, self.head_dim)
        
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
  
        out = F.scaled_dot_product_attention(q, k, v) 
        
        out = out.transpose(1, 2).reshape(N, L, C)
        out = self.proj(out)
        
        return out


class SinogramAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, H, W):
        super().__init__()
        self.row_attn = AxialAttention1D_RoPE(dim, num_heads, seq_len=W)
        self.row_norm = nn.LayerNorm(dim)
        
        self.col_attn = AxialAttention1D_RoPE(dim, num_heads, seq_len=H)
        self.col_norm = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x_row = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        x_row = x_row + self.row_attn(self.row_norm(x_row))
        x = x_row.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x_col = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        x_col = x_col + self.col_attn(self.col_norm(x_col))
        x = x_col.reshape(B, W, H, C).permute(0, 3, 2, 1)

        x_ffn = x.permute(0, 2, 3, 1) 
        x_ffn = x_ffn + self.ffn(x_ffn)
        x = x_ffn.permute(0, 3, 1, 2) 

        return x
    





class SparseSinogramFusionAttention(nn.Module):
    def __init__(self, dim, num_heads, H, W):
        super().__init__()
        self.sinogram_attn = SinogramAttentionBlock(dim, num_heads, H, W)


    def forward(self, y_true_feat):
        out_feat = self.sinogram_attn(y_true_feat)
        return out_feat


