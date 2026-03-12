import math
import argparse
from functools import partial
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

# from ray import tune
# from ray.air import Checkpoint, session
# from ray.tune.schedulers import ASHAScheduler

import models
from losses import DC_and_BCE_loss, PearsonCorrLoss
from dataloading import RegDataLoaderV1
from augmentation import get_moreDA_augmentation, default_3D_augmentation_params
from utils import *


class DiceCoeff(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = (y_true_f * y_pred_f).sum()
        return (2. * intersection + self.smooth) / (y_true_f.sum() + y_pred_f.sum() + self.smooth)

def plot_scatter(y_pred, y_true, epoch):
    ncols = y_pred.shape[-1]
    fig, axes = plt.subplots(1, ncols, figsize=(24, 6))
    axes = np.array([axes]).reshape(-1, )
    xx = np.linspace(-0.1, 1.1, 200)
    yy = xx
    for c in range(ncols):
        map = mean_absolute_error(y_pred[:, c], y_true[:, c])
        r, _ = pearsonr(y_pred[:, c], y_true[:, c])

        axes[c].scatter(y_pred[:, c], y_true[:, c])
        axes[c].set_title(f"Epoch {epoch}\n[MAE={map:.4f} Pearson r={r:.3f}]")
        axes[c].plot(xx, yy, '-r')
    
    axes[0].set_xlabel("Predicted Dice")
    axes[0].set_ylabel("True Dice")

    # fig.savefig(join(output_folder, "scatter.png"), dpi=300)
    # plt.close("all")
    return plt.gcf()

def train(args):
    fix_random_seeds(args.seed)
    writer = SummaryWriter(args.save_dir)

    ##### intialize dataloader
    patch_size = (160, 192, 160)
    batch_size = args.batch_size
    dataloader_tr = RegDataLoaderV1(args.dataroot, 'tr', args.fold, patch_size, batch_size, number_of_threads_in_multithreaded=default_3D_augmentation_params['num_threads'], infinite=True, concate_seg=True)
    dataloader_val = RegDataLoaderV1(args.dataroot, 'val', args.fold, patch_size, batch_size, number_of_threads_in_multithreaded=default_3D_augmentation_params['num_threads'], infinite=True, concate_seg=True)

    tr_gen, val_gen = get_moreDA_augmentation(dataloader_tr, dataloader_val, patch_size, default_3D_augmentation_params, pin_memory=True, use_nondetMultiThreadedAugmenter=True, seeds_val=[args.seed] * default_3D_augmentation_params['num_threads'])
    
    ##### intialize model
    arch = models.__dict__[args.arch]
    model = arch(num_input_channels=5)
    model = nn.DataParallel(model)
    # print(model)

    if args.compile_model:
        model = torch.compile(model)

    device = "cpu"
    if torch.cuda.is_available():
        print("GPU avaliable")
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
    device = torch.device(device)
    # print(device)
    model.to(device)

    if args.loss_func == "MSE":
        reg_criterion = nn.MSELoss().to(device)
    elif args.loss_func == "MAE":
        reg_criterion = nn.L1Loss().to(device)
    elif args.loss_func == "Huber":
        reg_criterion = nn.HuberLoss(delta=1).to(device)

    if args.use_pearson_loss:
        pearson_criterion = PearsonCorrLoss().to(device)

    seg_criterion = DC_and_BCE_loss({}, {'batch_dice': True, 'smooth': 1e-5, 'do_bg': True}).to(device)
    seg_metric = DiceCoeff()

    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0, verbose=True)

    if args.checkpoint and isfile(args.checkpoint):
        checkpoint_state = torch.load(args.checkpoint)
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint_state["lr_scheduler_state_dict"])
        start_epoch = 0
    else:
        start_epoch = 0

    if args.fp16:
        scaler = GradScaler()
    else:
        scaler = None

    best_r = 0.0
    best_mae = 10000
    print(scaler)

    for epoch in tqdm(range(start_epoch, args.max_epochs)):  # loop over the dataset multiple times
        ### resample the dataset every epoch
        tr_gen.generator.resample()
        
        ### training 
        running_loss = 0.0
        epoch_loss_tr = 0.0
        epoch_steps = 0

        train_nps = np.zeros((args.num_tr_batches * args.batch_size, 2), dtype=np.float32)

        model.train()
        optimizer.zero_grad()
        for batch_idx in range(args.num_tr_batches):
            batch = next(tr_gen)
            data = batch['data'].to(device)
            seg = batch['seg'].to(device)
            target = batch['target'].to(device)

            seg_pred = seg[:, 0, None, ...]
            seg_gt = seg[:, 1, None, ...]
            data = torch.cat([data, seg_pred], dim=1)
            target_mask = (seg_pred.long() != seg_gt.long()).float()

            with autocast(scaler is not None, dtype=torch.float16):
                if 'resunet' in args.arch:
                    output_reg, output_seg = model(data)
                    loss = reg_criterion(target, output_reg) + args.lmd * seg_criterion(output_seg, target_mask)
                else:
                    output_reg = model(data)
                    loss = reg_criterion(target, output_reg)

                if epoch > 5:
                    if args.use_pearson_loss:
                        loss += args.pearson_loss_weight * pearson_criterion(target, output_reg)

                loss = loss / args.iters_to_accumulate # Accumulates scaled gradients.

            target_np = target.detach().cpu().numpy().squeeze()
            output_reg_np = output_reg.detach().cpu().numpy().squeeze()
            train_nps[batch_idx*(args.batch_size):(batch_idx+1)*(args.batch_size), 0] = target_np
            train_nps[batch_idx*(args.batch_size):(batch_idx+1)*(args.batch_size), 1] = output_reg_np

            if scaler is None:
                loss.backward()
                if (batch_idx + 1) % args.iters_to_accumulate == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.iters_to_accumulate == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            # print statistics
            log_freq = args.num_tr_batches // 5
            running_loss += loss.item()
            epoch_loss_tr += loss.item()
            epoch_steps += 1
            if batch_idx % log_freq == log_freq - 1:  # print every log_freq mini-batches
                args.logger.info(f"[{batch_idx}/{args.num_tr_batches}] train loss:{running_loss / (epoch_steps):.3f}")
                writer.add_scalar('running losss', running_loss / (epoch_steps), epoch_steps)
                running_loss = 0.0

        train_mae = mean_absolute_error(train_nps[..., 0], train_nps[..., 1])
        train_r, _ = pearsonr(train_nps[..., 0], train_nps[..., 1])
        epoch_loss_tr /= args.num_tr_batches

        ### validation
        val_nps = np.zeros((args.num_val_batches * args.batch_size, 2), dtype=np.float32)
        model.eval()
        epoch_loss_val = 0.0
        epoch_dice_val = 0.0
    
        with torch.no_grad():
            for batch_idx in range(args.num_val_batches):
                batch = next(val_gen)
                data = batch['data'].to(device)
                seg = batch['seg'].to(device)
                target = batch['target'].to(device)

                seg_pred = seg[:, 0, None, ...]
                seg_gt = seg[:, 1, None, ...]
                data = torch.cat([data, seg_pred], dim=1)
                target_mask = (seg_pred.long() != seg_gt.long()).float()

                with autocast(scaler is not None):
                    if 'resunet' in args.arch:
                        output_reg, output_seg = model(data)
                        loss = reg_criterion(target, output_reg) + args.lmd * seg_criterion(output_seg, target_mask)
                    else:
                        output_reg = model(data)
                        loss = reg_criterion(target, output_reg) 

                    if epoch > 5:
                        if args.use_pearson_loss:
                            loss += args.pearson_loss_weight * pearson_criterion(target, output_reg)

                target_np = target.detach().cpu().numpy().squeeze()
                output_reg_np = output_reg.detach().cpu().numpy().squeeze()
                val_nps[batch_idx*(args.batch_size):(batch_idx+1)*(args.batch_size), 0] = target_np
                val_nps[batch_idx*(args.batch_size):(batch_idx+1)*(args.batch_size), 1] = output_reg_np
                epoch_loss_val += loss.item()
                
                if 'resunet' in args.arch:
                    pred_mask = torch.sigmoid(output_seg) >= 0.5
                    dice_err = seg_metric(pred_mask, target_mask)
                    epoch_dice_val += dice_err.item()

        val_mae = mean_absolute_error(val_nps[..., 0], val_nps[..., 1])
        val_r, _ = pearsonr(val_nps[..., 0], val_nps[..., 1])
        epoch_loss_val /= args.num_val_batches
        epoch_dice_val /= args.num_val_batches

        scatter = plot_scatter(val_nps[..., 1].reshape(-1, 1), val_nps[..., 0].reshape(-1, 1), epoch)

        writer.add_figure('scatter', scatter, epoch)
        writer.add_scalar('val Dice[err]', epoch_dice_val, epoch)
        writer.add_scalars('losss', {'train': epoch_loss_tr, 'val': epoch_loss_val}, epoch)
        writer.add_scalars('MAE', {'train': train_mae, 'val': val_mae}, epoch)
        writer.add_scalars('r-score', {'train': train_r, 'val': val_r}, epoch)
        
        if val_r > best_r:
            best_r = val_r
            best_dice = epoch_dice_val

            ### communicate with Ray Tune
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict()
            }   

            if scaler is not None:
                checkpoint_data['fp16_scaler'] = scaler.state_dict()

            torch.save(checkpoint_data, join(args.save_dir, "model_best_r.pt"))

        if val_mae < best_mae:
            best_mae = val_mae
            best_dice = epoch_dice_val

            ### communicate with Ray Tune
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict()
            }   

            if scaler is not None:
                checkpoint_data['fp16_scaler'] = scaler.state_dict()

            torch.save(checkpoint_data, join(args.save_dir, "model_best_mae.pt"))

        ### communicate with Ray Tune
        checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # "lr_scheduler_state_dict": lr_scheduler.state_dict()
        }   

        if scaler is not None:
            checkpoint_data['fp16_scaler'] = scaler.state_dict()

        torch.save(checkpoint_data, join(args.save_dir, "model_last.pt"))

        args.logger.info(f"========== [{epoch}/{args.max_epochs}] loss_tr:{epoch_loss_tr:.5f} loss_val:{epoch_loss_val:.5f}  best r-score:{best_r:.4f} best mae:{best_mae:.4f} best dice:{best_dice:.3f} ==========")

    ## destroy I/O 
    print("Finished Training")
    writer.close()
    del tr_gen, val_gen

if __name__ == "__main__":
    arch_chocies = ['resnet34', 'resnet50', 
                    'unet', 
                    'resunet34', 'resunet50', 
                    'resunet34_deeplabv3', 'resunet34_fusion', 
                    'resunet50_deeplabv3', 'resunet50_fusion', 
                    'resunet34_attn', 'resunet50_attn', 
                    'resunet34_inverted', 'resunet50_inverted']
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/mnt/beegfs/scratch/peijie.qiu/dataset/seg_reg_BraTS_cv')
    parser.add_argument("--save_dir", type=str, default='/mnt/beegfs/scratch/peijie.qiu/regression_v2')
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--num_tr_batches', default=1000, type=int)
    parser.add_argument('--num_val_batches', default=1000, type=int)
    ## model parameters
    parser.add_argument("--arch", type=str, default='resunet34', choices=arch_chocies)
    parser.add_argument("--loss_func", type=str, default='MAE', choices=['MSE', 'MAE', 'Huber'])
    parser.add_argument("--pearson_loss_weight", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lmd", type=float, default=2.0)
    parser.add_argument("--iters_to_accumulate", type=int, default=1)
    parser.add_argument("--disable_pearson_loss", action='store_true')
    parser.add_argument("--disable_fp16", action='store_true')
    parser.add_argument("--compile_model", action='store_true')
    # other parameters
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.save_dir = join(args.save_dir, args.arch)
    maybe_mkdir_p(args.save_dir)
    args.save_dir = make_dirs(args.save_dir)
    maybe_mkdir_p(args.save_dir)

    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    args.logger = get_logger(logging_path)
    args.fp16 = not args.disable_fp16
    args.use_pearson_loss = not args.disable_pearson_loss

    ######### save hyper parameters #########
    option = vars(args) ## args is the argparsing

    file_name = os.path.join(args.save_dir, 'hyper.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    print(args)

    train(args)
    

    

    
