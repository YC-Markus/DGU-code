import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import models.utils as utils


from models.DGU_model import DGU
from dataset_AAPM import *
from monai.metrics import PSNRMetric
from generative.metrics import SSIMMetric
from generative.losses import PerceptualLoss



DEVICE_ID       = 0
DEVICE          = torch.device(f"cuda:{DEVICE_ID}")
NUM_WORKERS     = 8
BATCH_SIZE      = 8
MAX_EPOCH       = 50
SAVE_EVERY_EPOCH = True
EMA_START_EPOCH = 4
EMA_RATE = 0.9995
LR              = 2e-4
TRAIN_SET_DIR   = Path("/data/HIT/hit0/yc/DATASET/SVCT/ldct/train_set") 
VAL_SET_DIR     = Path("/data/HIT/hit0/yc/DATASET/SVCT/ldct/val_set")
LOG_DIR         = Path("/data/HIT/hit0/yc/DGU/test")
num_views, detectors = 32, 256
radon_32x16 = utils.fanbeam_gen(sparsity=(num_views, num_views), img_size=detectors//16, bias=0, pixel_spacing=1.4285/16, det_spacing=1.2858*16)
radon_32x32 = utils.fanbeam_gen(sparsity=(num_views, num_views), img_size=detectors//8, bias=0, pixel_spacing=1.4285/8, det_spacing=1.2858*8)
radon_32x64 = utils.fanbeam_gen(sparsity=(num_views, num_views), img_size=detectors//4, bias=0, pixel_spacing=1.4285/4, det_spacing=1.2858*4)
radon_32x128 = utils.fanbeam_gen(sparsity=(num_views, num_views), img_size=detectors//2, bias=0, pixel_spacing=1.4285/2, det_spacing=1.2858*2)
radon_32x256 = utils.fanbeam_gen(sparsity=(num_views, num_views), img_size=detectors, bias=0)
radon_512x256 = utils.fanbeam_gen(sparsity=(512, 512), img_size=detectors, bias=0)
RADON = {16: radon_32x16, 32: radon_32x32, 64:radon_32x64, 128:radon_32x128, 256:radon_32x256, 512:radon_512x256}







LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE        = LOG_DIR / "test.log"
MODEL_DIR       = LOG_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("train-log")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, mode="a")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

psnr = PSNRMetric(max_val=1.)
ssim = SSIMMetric(spatial_dims=2, data_range=1.)
lpips = PerceptualLoss(spatial_dims=2, network_type="alex").to(DEVICE)


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    EVAL = {'SSIM_256':[], 'PSNR_256':[], 'LPIPS_256':[], 'L1_256':[], 'L2_256':[]}
    resolution = detectors

    for batch_idx, batch in enumerate(val_loader):
        ct_full = utils.pre_process(batch, val_flag=True, img_size=detectors, device=DEVICE)        
        pred_list = model(ct_full, RADON)

        img_pred  = pred_list[-1].clamp_(0, 1)
        img_gt    = ct_full.clamp_(0, 1)

        EVAL = utils.evaluate(img_pred, img_gt, ssim, psnr, lpips, EVAL, resolution=256)
        l1_sino = F.l1_loss(RADON[256].forward(img_pred), RADON[256].forward(ct_full), reduction='none').mean().item()
        EVAL[f'L1_{resolution}'][-1] = l1_sino
        
    ssim_mean, ssim_std = np.mean(EVAL[f'SSIM_{resolution}']), np.std(EVAL[f'SSIM_{resolution}'])
    psnr_mean, psnr_std = np.mean(EVAL[f'PSNR_{resolution}']), np.std(EVAL[f'PSNR_{resolution}'])
    lpips_mean, lpips_std = np.mean(EVAL[f'LPIPS_{resolution}']), np.std(EVAL[f'LPIPS_{resolution}'])
    l1_mean, l1_std = np.mean(EVAL[f'L1_{resolution}_SINO']), np.std(EVAL[f'L1_{resolution}'])
    l2_mean, l2_std = np.mean(EVAL[f'L2_{resolution}_IMG']), np.std(EVAL[f'L2_{resolution}'])
    return ssim_mean, psnr_mean, lpips_mean, l1_mean, l2_mean, ssim_std, psnr_std, lpips_std, l1_std, l2_std



def get_dataloaders():
    train_loader = DataLoader(CTDataset(TRAIN_SET_DIR), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(CTDataset(VAL_SET_DIR), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader



def main():
    torch.cuda.set_device(DEVICE_ID)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    logger.info(f"Using device: {DEVICE}")
    train_loader, val_loader = get_dataloaders()
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = DGU().to(DEVICE)


    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)

    def initialize_weights(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
    initialize_weights(model)


    max_steps = len(train_loader) * MAX_EPOCH
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=max_steps,
        eta_min=1e-6
    )




    best_ssim = -1.0
    loss_func = torch.nn.L1Loss()
    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        epoch_loss = 0.0
        if epoch == EMA_START_EPOCH: ema_model = utils.ModelEMA(model, decay=EMA_RATE)

        pbar = tqdm(enumerate(train_loader, start=1),
                    total=len(train_loader),
                    desc=f"Epoch {epoch}/{MAX_EPOCH}",
                    ncols=100)

        for batch_idx, batch in pbar:
            optimizer.zero_grad()
            loss = 0
            with torch.no_grad(): ct_full = utils.pre_process(batch, val_flag=False, img_size=detectors, device=DEVICE)

            pred_list1 = model(ct_full, RADON)

            loss = loss + loss_func(pred_list1[0], F.adaptive_avg_pool2d(ct_full, (32,32))) + loss_func(pred_list1[1], F.adaptive_avg_pool2d(ct_full, (32,32)))
            loss = loss + loss_func(pred_list1[2], F.adaptive_avg_pool2d(ct_full, (64,64))) + loss_func(pred_list1[3], F.adaptive_avg_pool2d(ct_full, (64,64))) + loss_func(pred_list1[4], F.adaptive_avg_pool2d(ct_full, (64,64)))
            loss = loss + loss_func(pred_list1[5], F.adaptive_avg_pool2d(ct_full, (128,128))) + loss_func(pred_list1[6], F.adaptive_avg_pool2d(ct_full, (128,128))) + loss_func(pred_list1[7], F.adaptive_avg_pool2d(ct_full, (128,128)))
            loss = loss + loss_func(pred_list1[8], ct_full) + loss_func(pred_list1[9], ct_full) + loss_func(pred_list1[10], ct_full) + loss_func(pred_list1[11], ct_full)
            loss.backward()
            optimizer.step()
            if epoch >= EMA_START_EPOCH: 
                ema_model.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{(epoch_loss/(batch_idx+1)):.4f}")

            scheduler.step()

        
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"[TRAIN] Epoch {epoch} | Avg L1 Loss: {avg_loss:.6f}")

        if epoch < EMA_START_EPOCH: 
            ssim_mean, psnr_mean, lpips_mean, l1_mean, l2_mean, ssim_std, psnr_std, lpips_std, l1_std, l2_std = validate(model, val_loader)
        else: 
            ssim_mean, psnr_mean, lpips_mean, l1_mean, l2_mean, ssim_std, psnr_std, lpips_std, l1_std, l2_std = validate(ema_model.get(), val_loader)
        val_msg = f"[VAL] Epoch {epoch} | PSNR: {psnr_mean:.5f}, SSIM: {ssim_mean:.5f}, LPIPS: {lpips_mean:.5f}, L1: {l1_mean:.5f}"
        logger.info(val_msg)
        val_msg = f"[VAL] Epoch {epoch} | PSNR_std: {psnr_std:.5f}, SSIM_std: {ssim_std:.5f}, LPIPS_std: {lpips_std:.5f}, L1_std: {l1_std:5f}"
        logger.info(val_msg)

        save_path_best = MODEL_DIR / "best.pth"
        if ssim_mean > best_ssim and epoch>=EMA_START_EPOCH:
            best_ssim = ssim_mean
            torch.save({
            'epoch': epoch,
            'model_state_dict': ema_model.get().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, save_path_best)
            logger.info(f"★ New best SSIM ({best_ssim:.4f}) → save to {save_path_best}")

        if SAVE_EVERY_EPOCH and epoch>=EMA_START_EPOCH:
            save_path_epoch = MODEL_DIR / f"epoch_{epoch:03d}.pth"
            torch.save({
            'epoch': epoch,
            'model_state_dict': ema_model.get().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, save_path_epoch)



    logger.info("Training finished.")


if __name__ == "__main__":
    main()
