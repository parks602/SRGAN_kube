import argparse, os
import numpy as np
import pandas as pd
import torch
from dataset import test_Dataset, test_datasets3d

from datetime import datetime, timedelta
from torch.autograd import Variable
from model import Generator
from torch.utils.data import DataLoader


def test(task, utc, ftime, args, date, device):

    input              = test_Dataset(args, var, ftime, date)
    test_dataset       = test_datasets3d(input)
    test_loader        = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False, num_workers=4)

    mask               = np.load(args.mask_dir)
    gis                = np.load(args.gis_dir)

    gis                = dataset.MinMaxscaler(0, 2600, gis)
    hight, landsea     = [], []
  
    for ii in range(test_conf['batch_size']):
        hight.append(gis)
        landsea.append(mask)
  
    hight, landsea     = np.asarray(hight), np.asarray(landsea)
    hight              = np.expand_dims(hight, axis=1)
    landsea            = np.expand_dims(landsea, axis=1)
  
    hight              = torch.as_tensor(hight, dtype=torch.float)
    landsea            = torch.as_tensor(landsea, dtype=torch.float)
  
    real_hight         = Variable(hight).to(device)
    real_landsea       = Variable(landsea).to(device)
  
    #best_loc = notebook['PSNR'].idxmax()
    model              = Generator(args.upscale_factor).eval()
    model.to(device)
    MODEL_NAME         = '%s/%s_%s_%s.pt'%(args.model_dir, task, utc, str(ftime.zfill(2)))

    model.load_state_dict(torch.load(MODEL_NAME, map_location=device))

    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.to(device)
            input = Variable(input)
            output = model(input, real_landsea, real_hight)
            output = output[0][0]
            if USE_CUDA:
                output = output.cpu()
            out = output.clone().detach().numpy()
  
    return out

def run(args):
    USE_CUDA           = torch.cuda.is_available()
    device             = torch.device('cuda' if USE_CUDA else 'cpu')
    print(device)

    for task in args.tasks:
        for utc in args.utcs:
            for ftime in args.ftimes:
                save_dir = '/%s/%s/%s/'%(task, utc, str(ftime.zfill(2)))
                if not os.path.exits(save_dir):
                    os.makedirs(save_dir)
                print('TEST start task : %s, utc : %s, ftime : %s'%(task, utc, str(ftime)))
                datelist = standardDate(args.sdate+utc, args.edate+utc)
                for date in datelist:
                    output = test(task, utc, ftime, args, date, device)
                    np.save('%s/%s.npy'%(save_dir, date), output)
                    print('TASK : %s, UTC : %s, ftime : %s, %s is saved'%(task, utc, str(ftime), date)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--fmt', type=str, default = '%Y%m%d%H')
    parser.add_argument('--x_dir', type=str, default = '/home/pkw/2022/DATA/data/x')
    parser.add_argument('--y_dir', type=str, default = '/home/pkw/2022/DATA/data/y')
    parser.add_argument('--tasks', type=list, default = ['T3H', 'REH'])
    parser.add_argument('--utcs',  type=list, default = ['00', '12'])
    parser.add_argument('--ftimes', type=list, default = [2,3])
    parser.add_argument('--sdate', type=str, default = '20210301')
    parser.add_argument('--edate', type=str, default = '20210330')
    parser.add_argument('--mask_dir', type=str, default = '/home/pkw/2022/srgan/DABA/noaa_lsm1km.npy')
    parser.add_argument('--gis_dir', type=str, default = '/home/pkw/2022/srgan/DABA/output_elev_1KM_Mean_SRTM.npy')

    parser.add_argument('--batch_szie', type=int, default = 1)
    parser.add_argument('--upscale_factor', type=int, default = 5)
    parser.add_argument('--model_dir', type=str, default = '/home/pkw/2022/srgan/model/')

