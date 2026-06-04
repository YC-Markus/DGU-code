import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft




class StructureGating(nn.Module):
    """
    Structure(Phys) -> Texture(Perc)
    """
    def __init__(self, ch_struc, ch_tex):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(ch_struc, ch_tex, kernel_size=3, padding=1), 
            nn.GELU(),
            nn.Conv2d(ch_tex, ch_tex, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_struc, x_tex):
        gate_map = self.gate_conv(x_struc)
        x_tex_guided = x_tex * gate_map
        return x_tex_guided

class TextureSFT(nn.Module):
    """
    Texture(Perc) -> Structure(Phys)
    """
    def __init__(self, ch_struc, ch_tex):
        super().__init__()
        self.sft_net = nn.Sequential(
            nn.Conv2d(ch_tex, ch_struc, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch_struc, ch_struc * 2, kernel_size=1) 
        )
        nn.init.constant_(self.sft_net[-1].weight, 0)
        nn.init.constant_(self.sft_net[-1].bias, 0)

    def forward(self, x_struc, x_tex):
        sft_params = self.sft_net(x_tex)
        gamma, beta = torch.chunk(sft_params, 2, dim=1)
        x_struc_enhanced = x_struc * (1 + gamma) + beta
        return x_struc_enhanced





class APCM(nn.Module):
    """
    Amplitude-Phase Cross-Modulation
    """
    def __init__(self, ch_phys, ch_perc):
        super().__init__()
        
        in_ch_phase = (ch_phys + ch_perc) * 2
        self.phase_mlp = nn.Sequential(
            nn.Conv2d(in_ch_phase, ch_perc, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ch_perc, ch_perc, kernel_size=3, padding=1), 
        )
        nn.init.constant_(self.phase_mlp[-1].weight, 0)
        nn.init.constant_(self.phase_mlp[-1].bias, 0)


        in_ch_amp = ch_phys + ch_perc
        self.amp_mlp = nn.Sequential(
            nn.Conv2d(in_ch_amp, ch_phys, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ch_phys, ch_phys, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        self.lambda_phys = nn.Parameter(torch.zeros(1, ch_phys, 1, 1))
        self.lambda_perc = nn.Parameter(torch.zeros(1, ch_perc, 1, 1))

    def forward(self, x_phys, x_perc):
        F_phys = torch.fft.rfft2(x_phys, norm='ortho') 
        F_perc = torch.fft.rfft2(x_perc, norm='ortho')

        A_phys = torch.abs(F_phys) + 1e-6
        A_perc = torch.abs(F_perc) + 1e-6

        phys_real_norm = F_phys.real / A_phys
        phys_imag_norm = F_phys.imag / A_phys
        perc_real_norm = F_perc.real / A_perc
        perc_imag_norm = F_perc.imag / A_perc

        concat_feat = torch.cat([phys_real_norm, phys_imag_norm, perc_real_norm, perc_imag_norm], dim=1)
        delta_P = self.phase_mlp(concat_feat)
        
        unit_rotation = torch.complex(torch.cos(delta_P), torch.sin(delta_P))
        F_perc_new = F_perc * unit_rotation
        
        A_perc_log = torch.log(A_perc + 1e-4)
        A_phys_log = torch.log(A_phys + 1e-4)
        M = self.amp_mlp(torch.cat([A_perc_log, A_phys_log], dim=1))

        F_phys_new = F_phys * M.to(F_phys.dtype)
        
        x_phys_rec = torch.fft.irfft2(F_phys_new, norm='ortho', s=x_phys.shape[-2:])
        x_perc_rec = torch.fft.irfft2(F_perc_new, norm='ortho', s=x_perc.shape[-2:])
        
        out_phys_freq = x_phys + self.lambda_phys * x_phys_rec
        out_perc_freq = x_perc + self.lambda_perc * x_perc_rec
        
        return out_phys_freq, out_perc_freq





class FSI(nn.Module):
    """
    Bi-directional Interaction
    """
    def __init__(self, ch_phys, ch_perc):
        super().__init__()
        self.c_freq_phys = ch_phys // 4
        self.c_spatial_phys = ch_phys - self.c_freq_phys
        
        self.c_freq_perc = ch_perc // 4
        self.c_spatial_perc = ch_perc - self.c_freq_perc
        
        self.structure_gating = StructureGating(self.c_spatial_phys, self.c_spatial_perc)
        self.texture_sft = TextureSFT(self.c_spatial_phys, self.c_spatial_perc)
        self.apcm = APCM(self.c_freq_phys, self.c_freq_perc)
        
        self.fuse_phys = nn.Sequential(
            nn.Conv2d(ch_phys, ch_phys, kernel_size=1),
            nn.GELU(), 
            nn.Conv2d(ch_phys, ch_phys, kernel_size=3, padding=1)
        )
        self.fuse_perc = nn.Sequential(
            nn.Conv2d(ch_perc, ch_perc, kernel_size=1),
            nn.GELU(), 
            nn.Conv2d(ch_perc, ch_perc, kernel_size=3, padding=1)
        )
        nn.init.constant_(self.fuse_phys[-1].weight, 0)
        nn.init.constant_(self.fuse_phys[-1].bias, 0)
        nn.init.constant_(self.fuse_perc[-1].weight, 0)
        nn.init.constant_(self.fuse_perc[-1].bias, 0)
        
    def forward(self, x_phys, x_perc):
        phys_freq, phys_spatial = torch.split(x_phys, [self.c_freq_phys, self.c_spatial_phys], dim=1)
        perc_freq, perc_spatial = torch.split(x_perc, [self.c_freq_perc, self.c_spatial_perc], dim=1)
        
        perc_spatial_out = self.structure_gating(phys_spatial, perc_spatial)
        phys_spatial_out = self.texture_sft(phys_spatial, perc_spatial)
        phys_freq_out, perc_freq_out = self.apcm(phys_freq, perc_freq)
        
        cat_phys = torch.cat([phys_freq_out, phys_spatial_out], dim=1)
        cat_perc = torch.cat([perc_freq_out, perc_spatial_out], dim=1)
        
        out_phys = self.fuse_phys(cat_phys) + cat_phys
        out_perc = self.fuse_perc(cat_perc) + cat_perc
        
        return out_phys, out_perc

