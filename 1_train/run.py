#1. train

import sys
import os
import logging
import argparse
import math 

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

from dataset import DatasetMaker, MinMaxscaler
from loss import GeneratorLoss
from model import Generator, Discriminator
from pytorchtools import EarlyStopping
from torch.cuda.amp import GradScaler, autocast


def train_model(task, utc, ftime, args, device):
    model_dir      = args.model_dir
    batch_size     = args.batch_size
    UPSCALE_FACTOR = args.upscale_factor
    NUM_EPOCHS     = args.epochs
    es_patience    = args.patience

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_save_name = '%s/%s_%s_%s.pt'%(model_dir, task, utc, str(ftime.zfill(2)))
    early_stopping    = EarlyStopping(patience=es_patience, verbose=True, path= model_save_name)

    ################################################################################
    '''MAKE DATASET'''
    ################################################################################
    mask              = np.load(args.mask_dir)
    gis               = np.load(args.gis_dir)
    gis               = MinMaxscaler(0, 2600, gis)
    hight, landsea    = [], []

    for ii in range(batch_size):
        hight.append(gis)
        ladnsea.append(mask)

    hight, landsea    = np.asarray(hight), np.asarray(landsea)

    hight             = np.expand_dims(hight, axis=1)
    landsea           = np.expand_dims(landsea, axis=1)

    hight             = torch.as_tensor(hight, dtype=torch.float)
    landsea           = torch.as_tensor(landsea, dtype=torch.float)

    real_hight        = Variable(hight).to(device)
    real_landsea      = Variable(landsea).to(device)

    fake_label        = torch.full((batch_size, 1), 0, dtype=hight.dtype).to(device)
    real_label        = torch.full((batch_size, 1), 1, dtype=hight.dtype).to(device)

    train_loader, valid_loader  = DatasetMaker(var, utc, ftime, args)


    ################################################################################
    '''MAKE MODEL'''
    ################################################################################
    netG              = Generator(UPSCALE_FACTOR)
    netD              = Discriminator()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    generator_criterion     = GeneratorLoss()
    discriminator_criterion = nn.BCEWithLogitsLoss().to(device)

    scalerD           = GradScaler()
    scalerG           = GradScaler()

    netG.to(device)
    netD.to(device)
    generator_criterion.to(device)
    discriminator_criterion.to(device)

    optimizerG        = optim.Adam(netG.parameters())
    optimizerD        = optim.Adam(netD.parameters())

    schedulerG        = optim.lr_scheduler.OneCycleLR(optimizerG,
                                                      max_lr = 0.001,
                                                      pct_start = 0.1,
                                                      epochs = NUM_EPOCHS,
                                                      steps_per_epoch = len(train_loader),\
                                                      anneal_strategy='linear')
    schedulerD        = optim.lr_scheduler.OneCycleLR(optimizerD,
                                                      max_lr = 0.001,
                                                      pct_start = 0.1,
                                                      epochs = NUM_EPOCHS,
                                                      steps_per_epoch = len(train_loader),
                                                      anneal_strategy='linear')

    results            = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'val_loss': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            if batch_size != train_conf['batch_size']:
                continue

            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img         = Variable(target).to(device)
            real_img         = real_img.to(device)
            z                = Variable(data).to(device)

            fake_img = netG(z, real_landsea, real_hight)

            netD.zero_grad()
            with autocast():
                real_out     = discriminator_criterion(netD(real_img), real_label)
                fake_out2    = discriminator_criterion(netD(fake_img.detach()), fake_label)
                d_loss       = real_out + fake_out2

            scalerD.scale(d_loss).backward(retain_graph=True)
            scalerD.step(optimizerD)
            scalerD.update()
            schedulerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            with autocast():
                fake_img     = netG(z, real_landsea, real_hight)
                fake_out     = netD(fake_img).mean()
                g_loss       = generator_criterion(fake_out, fake_img, real_img)

            scalerG.scale(g_loss).backward()
            scalerG.step(optimizerG)
            scalerG.update()
            schedulerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            train_bar.set_description(desc='[%d/%d] Loss_D:%.2f Loss_G:%.2f' % (
                epoch, NUM_EPOCHS,
                running_results['d_loss'] / running_results['batch_sizes'],
                math.sqrt(running_results['g_loss'] / running_results['batch_sizes']),
                ))

        netG.eval()

        if epoch%2 == 1:
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'val_loss': 0, 'batch_sizes': 0}
                val_images = []
                for val_data, val_target in val_bar:
                    batch_size = val_data.size(0)
                    if batch_size != train_conf['batch_size']:
                        continue
                    valing_results['batch_sizes'] += batch_size
                    lr = val_data.to(device)
                    hr = val_target.to(device)
                    sr  = netG(lr, real_landsea, real_hight)

                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse.item() * batch_size
                    val_bar.set_description(
                        desc='[converting LR images to SR images] Val_loss: %.4f' % (
                            math.sqrt(valing_results['mse'] /  valing_results['batch_sizes'])))

        # save model parameters
            early_stopping(math.sqrt(valing_results['mse']/ valing_results['batch_sizes']), netG)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['val_loss'].append(valing_results['mse']/ valing_results['batch_sizes'])
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'],
                  'Val_loss': results['val_loss'] },
            index=range(1, epoch + 1))



def run(args):
    USE_CUDA     = torch.cuda.is_available()
    device       = torch.device('cuda' if USE_CUDA else 'cpu')
    print(device)

    for task in args.tasks:
        for utc in args.utcs:
            for ftime in args.ftimes:
                print('TRAIN start task : %s, utc : %s, ftime : %s'%(task, utc, str(ftime)))
                train_model(task, utc, ftime, args, device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fmt', type=str, default = '%Y%m%d%H')
    parser.add_argument('--x_dir', type=str, default = '/home/pkw/2022/DATA/data/x')
    parser.add_argument('--y_dir', type=str, default = '/home/pkw/2022/DATA/data/y')
    parser.add_argument('--mask_dir', type=str, default = '/home/pkw/2022/srgan/DABA/noaa_lsm1km.npy')
    parser.add_argument('--gis_dir', type=str, default = '/home/pkw/2022/srgan/DABA/output_elev_1KM_Mean_SRTM.npy')
    parser.add_argument('--tasks', type=list, default = ['T3H', 'REH'])
    parser.add_argument('--utcs',  type=list, default = ['00', '12'])
    parser.add_argument('--ftimes', type=list, default = [2,3])
    parser.add_argument('--sdate', type=str, default = '20210301')
    parser.add_argument('--edate', type=str, default = '20210330')

    parser.add_argument('--batch_szie', type=int, default = 2)
    parser.add_argument('--upscale_factor', type=int, default = 5)
    parser.add_argument('--epochs', type=int, default = 500)
    parser.add_argument('--patience', type=int, default = 50)
    parser.add_argument('--model_dir', type=str, default = '/home/pkw/2022/srgan/model/')

    args  = parser.parse_args()
