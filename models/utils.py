import torch
import copy
import numpy as np
from torch_radon import Radon, RadonFanbeam
import torchvision.transforms.v2 as v2
import torch.nn.functional as F

def recon(projections, radon): 
    sino = projections
    filtered_sinogram = radon.filter_sinogram(sino, filter_name='ramp')
    fbp = radon.backprojection(filtered_sinogram)
    return fbp


def get_residual_image(img, sino, radon_layer):
    sino_est = radon_layer.forward(img)
    res_sino = sino - sino_est
    res_img = recon(res_sino, radon=radon_layer)
    return res_img


def CG_backable(sino, ct, numStore, radon):
    b, c, h, w = ct.shape
    device = ct.device

    numStore = set(numStore)
    numIter = max(numStore)

    conjGradRestart = 50
    correction_interval = 5
    eps = 1e-10

    x = torch.clamp(ct, min=0)
    Pf = radon.forward(x)

    Pf_dot_Pf = torch.sum(Pf * Pf, dim=(2, 3), keepdim=True)
    sino_dot_Pf = torch.sum(sino * Pf, dim=(2, 3), keepdim=True)

    cond_mask = (Pf_dot_Pf > 1e-8) & (sino_dot_Pf > 1e-8)
    scale_factor = torch.where(
        cond_mask,
        sino_dot_Pf / (Pf_dot_Pf + eps),
        torch.ones_like(Pf_dot_Pf),
    )

    x = x * scale_factor
    Pf = Pf * scale_factor
    residual = Pf - sino

    grad_old = torch.zeros_like(x)
    d = torch.zeros_like(x)
    grad_old_dot_grad_old = torch.zeros(b, c, 1, 1, device=device, dtype=x.dtype)

    CT = []

    for n in range(numIter):
        iter_idx = n + 1
        grad = radon.backprojection(residual)

        if n == 0 or (n % conjGradRestart) == 0:
            d = grad
        else:
            numerator = torch.sum(grad * (grad - grad_old), dim=(2, 3), keepdim=True)
            gamma = numerator / (grad_old_dot_grad_old + eps)
            gamma = torch.clamp(gamma, min=0.0) 
            d = gamma * d + grad

            d_dot_g = torch.sum(d * grad, dim=(2, 3), keepdim=True)
            reset_mask = d_dot_g <= 0
            if reset_mask.any():
                d = torch.where(reset_mask, grad, d)

        grad_old_dot_grad_old = torch.sum(grad * grad, dim=(2, 3), keepdim=True)
        grad_old = grad

        Pd = radon.forward(d)
        d_dot_g = torch.sum(d * grad, dim=(2, 3), keepdim=True)
        Pd_dot_Pd = torch.sum(Pd * Pd, dim=(2, 3), keepdim=True)

        valid = (d_dot_g > 0) & (Pd_dot_Pd > eps)
        stepSize = torch.where(
            valid,
            d_dot_g / (Pd_dot_Pd + eps),
            torch.zeros_like(d_dot_g),
        )

        x = x - stepSize * d
        x = torch.clamp(x, min=0)

        residual = residual - stepSize * Pd

        if (iter_idx % correction_interval == 0) or (iter_idx == numIter):
            residual = radon.forward(x) - sino

        if iter_idx in numStore:
            CT.append(x.clone()) 
    return torch.cat(CT, dim=1) if CT else ct














def fanbeam_gen(sparsity, img_size, bias=0, det_spacing=1.2858, pixel_spacing=1.4285, det_count=None):
    source_det_distance=1085.6
    source_distance=595.0
    if det_count == None: det_count = img_size    
    
    (sparse_scan, full_scan) = sparsity
    index = full_scan // sparse_scan
    assert full_scan % sparse_scan == 0
    angles = np.linspace(0, 2 * np.pi, full_scan, endpoint=False)
    seq = np.arange(0, full_scan, 1)
    if bias < 0: bias = len(seq) + bias
    seq = np.roll(seq, -bias)
    result_seq = seq[::index]
    angles = angles[result_seq]
    
    ops_example = RadonFanbeam(resolution=img_size, angles=angles, source_distance=source_distance*pixel_spacing,
                               det_distance=(source_det_distance-source_distance) * pixel_spacing,
                               det_count=det_count, det_spacing=det_spacing * pixel_spacing, clip_to_circle=True)
    return ops_example




transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5), v2.RandomRotation(degrees=180, interpolation=v2.InterpolationMode.BILINEAR)])
def pre_process(batch, val_flag, img_size, device):
    img = batch["image"].to(device)
    if not val_flag: 
        img = transforms(img)
        img = F.adaptive_avg_pool2d(img, (img_size, img_size))
    else:
        img = F.adaptive_avg_pool2d(img, (img_size, img_size))
    return img



def evaluate(ct_pred, ct_ref, ssim, psnr, lpips, EVAL, resolution):
    ct_pred, ct_ref = torch.clip(ct_pred, min=0, max=1), torch.clip(ct_ref, min=0, max=1)
    EVAL[f'SSIM_{resolution}'].append(ssim(ct_pred, ct_ref).mean().item())
    EVAL[f'PSNR_{resolution}'].append(psnr(ct_pred, ct_ref).mean().item())
    EVAL[f'LPIPS_{resolution}'].append(lpips(ct_pred, ct_ref).mean().item())
    EVAL[f'L1_{resolution}'].append(F.l1_loss(ct_pred, ct_ref, reduction='none').mean().item())
    EVAL[f'L2_{resolution}'].append(F.mse_loss(ct_pred, ct_ref, reduction='none').mean().item())
    return EVAL




class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema_model = copy.deepcopy(model).eval() 
        self.decay = decay
    def update(self, model):
        with torch.no_grad():
            for ema_params, model_params in zip(self.ema_model.parameters(), model.parameters()):
                ema_params.data *= self.decay
                ema_params.data += (1.0 - self.decay) * model_params.data
    def get(self):
        self.ema_model.eval()
        return self.ema_model