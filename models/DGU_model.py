import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import recon, CG_backable, get_residual_image
from models.RSD import SinogramDecomposition
from models.backbone import *
from models.FSI import FSI



class ChannelAdapter(nn.Module):
    def __init__(self, in_ch, out_ch, up=True):
        super().__init__()
        if up:
            self.net = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=4),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) 
            )
            
    def forward(self, x):
        return self.net(x)




class DUSTNet(nn.Module):
    def __init__(self, in_ch_a, in_ch_b, mid_channel_phys, mid_channel_perc, out_res, last_output=True, use_attn=True):
        super().__init__()
        self.last_output = last_output
        self.out_res = out_res

        if use_attn:
            self.feat_extractor_phys = nn.Sequential(nn.Conv2d(in_ch_a, mid_channel_phys, 1), NAFBlock_Phys(mid_channel_phys))
        else:
            self.feat_extractor_phys = nn.Sequential(nn.Conv2d(in_ch_a, mid_channel_phys, 1), NAFBlock_Perc(mid_channel_phys))
        
        self.feat_extractor_perc = nn.Sequential(nn.Conv2d(in_ch_b, mid_channel_perc, 1), NAFBlock_Perc(mid_channel_perc))

        self.bim = FSI(ch_phys=mid_channel_phys, ch_perc=mid_channel_perc)
        
        if self.last_output:
            self.head_A = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(mid_channel_phys, mid_channel_phys, kernel_size=3, padding=1, groups=4),
                nn.GELU(), nn.Conv2d(mid_channel_phys, 1, 3, 1, 1)
            )
            nn.init.constant_(self.head_A[-1].weight, 0)
            nn.init.constant_(self.head_A[-1].bias, 0)
        else:
            self.head_A = nn.Conv2d(mid_channel_phys, 1, 3, 1, 1)
            nn.init.constant_(self.head_A.weight, 0)
            nn.init.constant_(self.head_A.bias, 0)


    def forward(self, phys_in, perc_in):
        feat_phys_in = self.feat_extractor_phys(phys_in)
        feat_perc_in = self.feat_extractor_perc(perc_in)

        phys_fused, perc_fused = self.bim(feat_phys_in, feat_perc_in)
        
        if self.last_output:
            img_A = self.head_A(phys_fused) + F.interpolate(phys_in[:,-1,:,:].unsqueeze(1), scale_factor=2, mode='bilinear')
            return img_A, perc_fused
        else:
            img_A = self.head_A(phys_fused) + phys_in[:,-1,:,:].unsqueeze(1)
            return img_A, perc_fused





class DGU(nn.Module):
    def __init__(self, perc_channels=[48,64,96,32,32], mid_channels=[32,96,64,48], mid_channels_phys = [32,96,48,32], H=32, resolutions=[32, 64, 128, 256], blocks_per_stage=3, TCG=6):
        super().__init__()
        self.TCG = TCG
        self.resolutions = resolutions
        self.stages = nn.ModuleList()
        self.blocks_per_stage = blocks_per_stage
        self.sino_decomposition = SinogramDecomposition(perc_channels=perc_channels, H=H, resolutions=self.resolutions)
        
        
        for i, res in enumerate(self.resolutions):
            stage_perc_ch = perc_channels[-2 - i]
            mid_ch_perc = mid_channels[i]
            mid_ch_phys = mid_channels_phys[i] 
            in_ch_perc = perc_channels[-1] if i == 0 else mid_channels[i-1]

            is_last_stage = (i == len(self.resolutions) - 1)
            adapters = nn.ModuleList()
            for b in range(blocks_per_stage):
                up = (b == 0)
                adapters.append(ChannelAdapter(in_ch=in_ch_perc if up else mid_ch_perc, out_ch=stage_perc_ch, up=up))

            dustnets = nn.ModuleList()
            for b in range(blocks_per_stage):
                if is_last_stage:
                    dustnet = DUSTNet(in_ch_a=2+TCG*2, in_ch_b=stage_perc_ch*2, mid_channel_phys=mid_ch_phys, mid_channel_perc=mid_ch_perc, out_res=res, last_output=False, use_attn=False)
                else:
                    is_last_block = (b == blocks_per_stage - 1)
                    dustnet = DUSTNet(in_ch_a=2+TCG*2, in_ch_b=stage_perc_ch*2, mid_channel_phys=mid_ch_phys, mid_channel_perc=mid_ch_perc, out_res=res, last_output=is_last_block)
                dustnets.append(dustnet)

            self.stages.append(nn.ModuleDict({
                'adapters': adapters,
                'dustnets': dustnets
            }))


    def forward(self, ct_full, RADON):
        sinogram = RADON[self.resolutions[-1]].forward(ct_full)
        phys_sinos, perc_sinos = self.sino_decomposition(sinogram)
        
        # Init
        img_curr = recon(phys_sinos[16], radon=RADON[16])
        perc_curr = recon(perc_sinos[16], radon=RADON[16])
        
        out_list = []
        # Upsample initial img to match the first stage iteration
        img_curr = F.interpolate(img_curr, size=(self.resolutions[0], self.resolutions[0]), mode='bilinear', align_corners=False)
        
        for i, res in enumerate(self.resolutions):
            stage_modules = self.stages[i]
            phys_sino = phys_sinos[res]
            perc_sino = perc_sinos[res]
            radon_layer = RADON[res]
            
            for b in range(self.blocks_per_stage):
                adapter = stage_modules['adapters'][b]
                mixnet = stage_modules['dustnets'][b]
                
                # --- Physics Stream ---
                phys_remain = get_residual_image(img_curr, phys_sino, radon_layer)
                if self.TCG == 0:
                    phys_in = torch.cat([phys_remain, img_curr], dim=1)
                else:
                    phys_cg_trajectory = CG_backable(sino=phys_sino, ct=img_curr, numStore=[i for i in range(0,self.TCG+1)], radon=radon_layer)
                    phys_cg_trajectory_res = phys_cg_trajectory - img_curr
                    phys_in = torch.cat([img_curr, phys_remain, phys_cg_trajectory_res, phys_cg_trajectory], dim=1)
                # --- Perception Stream ---
                perc_feat = adapter(perc_curr)
                perc_remain = get_residual_image(perc_feat, perc_sino, radon_layer)
                perc_in = torch.cat([perc_feat, perc_remain], dim=1)
                
                # --- Fusion ---
                img_curr, perc_curr = mixnet(phys_in, perc_in)
                out_list.append(img_curr)                
        return out_list