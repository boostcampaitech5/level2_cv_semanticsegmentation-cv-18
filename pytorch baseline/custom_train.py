# python native
import os
import json
import time
import wandb
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A
import segmentation_models_pytorch as smp

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from argparse import ArgumentParser

# baseline
from utils import dice_coef, save_model, set_seed, AverageMeter
from dataset import XRayDataset, XRayTrainDataset, init_transform, CutMixCollator, CutMixCriterion
from model import init_models
from inference import inference
from loss import init_loss


def validation(epoch, model, data_loader, criterion, wandb_off, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)

                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    # log dice score per classes
    if not wandb_off:
        wandb.log(
            {
                c : d.item()
                for c, d in zip(CLASSES, dices_per_class)
            }
        )
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice


def train(saved_dir, exp_name, max_epoch, model, data_loader, val_loader, val_every, train_criterion, val_criterion, optimizer, lr_scheduler, wandb_off, mixed):
    print(f'Start training..')
    data_time = AverageMeter()
    n_class = len(CLASSES)
    best_dice = 0.
    
    if mixed:
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    end = time.time()
    for epoch in range(max_epoch):
        model.train()

        for step, (images, masks) in enumerate(data_loader):   
            data_time.update(time.time() - end)
         
            # images, masks = images.cuda(), masks.cuda()
            images = images.cuda()

            if isinstance(masks, (tuple, list)):
                masks1, masks2, lam = masks
                masks = (masks1.cuda(), masks2.cuda(), lam)
            else:
                masks = masks.cuda()

            model = model.cuda()

            optimizer.zero_grad()

            ###### mixed precision ######
            if mixed:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = train_criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()        
            #############################    
            
            else:
                outputs = model(images)
                ########### HRNetOCR #########
                if isinstance(outputs, list):
                    outputs = outputs[1]
                ##############################

                loss = train_criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            
            
            
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{max_epoch}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)},'
                    f'Data Time: {data_time.avg}'
                )

                if not wandb_off:
                    wandb.log(
                        {
                            "Train Loss" : loss
                        }
                    )
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        if (epoch + 1) % val_every == 0:
            dice = validation(epoch + 1, model, val_loader, val_criterion, wandb_off)
            
            if not wandb_off:
                wandb.log(
                    {
                        "Val Dice" : dice
                    }
                )
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {saved_dir}")
                best_dice = dice
                save_model(model, saved_dir, file_name=f"{exp_name}_best_model.pt")


def main(saved_dir, args):
    set_seed()

    if not args.wandb_off:
        wandb.init(
            project='semantic-segmentation',
            entity='cv-18',
            name=f'{args.exp_name}',
        )
    
    tf_train = init_transform(args.aug) 
    tf_val = init_transform('base2') 
    
    ################# if copy paste ##############
    if args.copypaste:
        train_dataset = XRayTrainDataset(is_train=True, transforms=tf_train, k=args.k)
        print(f'Copy Paste applied : {args.copypaste}, k : {args.k}')
    else:
        train_dataset = XRayDataset(is_train=True, transforms=tf_train) 
    valid_dataset = XRayDataset(is_train=False, transforms=tf_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=False, num_workers=args.val_num_workers, drop_last=False)

    model = init_models(args.model, args.encoder)

    train_criterion = init_loss(args.loss)
    val_criterion = init_loss(args.loss)

    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.optimizer](model.parameters(), lr=args.learning_rate)

    if args.lr_scheduler is not None:
        lr_scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[args.lr_scheduler](optimizer, T_max=args.max_epoch)
    else:
        lr_scheduler = None

    train(saved_dir, args.exp_name, args.max_epoch, model, train_loader, valid_loader, args.val_every, train_criterion, val_criterion, optimizer, lr_scheduler, args.wandb_off, args.mixed)


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument("--exp_name", type=str, default="318_deeplabv3plus_r101_dice_resize1024")
    parser.add_argument("--wandb_off", default=None) # None -> wandb on , True -> wandb off

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mixed', default=None) # mixed precision
    parser.add_argument('--copypaste', default=True) # copypaste augmentation
    parser.add_argument('--k', type=int, default=3)

    parser.add_argument('--model', type=str, default='deeplabv3')
    parser.add_argument('--encoder', type=str, default='r101')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss', type=str, default='dice')
    parser.add_argument('--aug', type=str, default='base2')
    parser.add_argument('--lr_scheduler', default=None) # 'CosineAnnealingLR'

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--val_num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_every', type=int, default=1) 
    parser.add_argument('--learning_rate', type=float, default=1e-4) 
    parser.add_argument('--max_epoch', type=int, default=50) 

    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = parse_args()

    saved_dir = f"../checkpoints/result_{args.exp_name}/"
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)
    
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]

    main(saved_dir, args)
    inference(saved_dir, args.exp_name)