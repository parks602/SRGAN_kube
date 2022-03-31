import sys, os, warnings, warnings
import numpy as np
import torch
import pygrib
import random
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import ReadASCII
warnings.filterwarnings(action='ignore')

def RandomDate(sdate, edate):
    random.seed(1)
    fmt = "%Y%m%d%H"
    dt_sdate = datetime.strptime(sdate, fmt)  ### str -> datetime
    dt_edate = datetime.strptime(edate, fmt)
    day_list = []
    now = dt_sdate
  
    while now<=dt_edate:
        ex_sdate = now.strftime(fmt)
        day_list.append(ex_sdate)
        now = now + timedelta(days=1)
    train_list = sorted(random.sample(day_list, int(len(day_list)*6//10)))
  
    for i in range(len(train_list)):
        day_list.remove(train_list[i])
        valid_list = day_list
  
    test_list  = sorted(random.sample(valid_list, int(len(valid_list)*5//10)))
  
    for j in range(len(test_list)):
        valid_list.remove(test_list[j])
  
    return (train_list, valid_list, test_list)


def standardDate(sdate, edate):
    fmt = "%Y%m%d%H"
    dt_sdate = datetime.strptime(sdate, fmt)  ### str -> datetime
    dt_edate = datetime.strptime(edate, fmt)
    day_list = []
    now = dt_sdate
  
    while now <= dt_edate:
        ex_sdate = now.strftime(fmt)
        day_list.append(ex_sdate)
        now = now + timedelta(days=1)
    return day_list


def FileExists(path):
    if not os.path.exists(path):
        print("Can't Find : %s" %(path))
        return False
    else:
        return True


def MinMaxscaler(Min, Max, data):
    minmax = (data - Min)/(Max - Min)
    return(minmax)

def MakeDataset(args, var, ftime, date_list):
    #=== Config
    x_dir  = args.x_dir
    y_dir  = args.y_dir
    ftime  = int(ftime)
    fmt    = args.fmt
    #=== Read Data
    xdata, temp, dates = [],[], []
    for date in date_list:
        ldate = datetime.strptime(date, fmt)
        ldate = ldate + timedelta(hours=ftime*3)
        ldate = datetime.strftime(ldate,fmt)

        x_name   = '%s/%s_%s' %(x_dir var, ldate))
        y_name   = "%s/%s_%s" %(y_dir, var, ldate) #ANAL
        if not FileExists(x_name) or not FileExists(y_name):
            print(date, 'is not exist')
            continue
        xdat = np.load(x_name)
        ydat = np.load(y_name)
        if var == "REH":
            xdat[np.where(xdat>99.9)]=99.9
            xdat = MinMaxscaler(0, 100, xdat)
        elif var == "T3H":
            xdat = MinMaxscaler(-50, 50, xdat)
        xdata.append(xdat)
        temp.append(ydat)
        dates.append(date)
    xdata = np.asarray(xdata)
    temp = np.asarray(temp)
    return(xdata, temp, dates)

def test_Dataset(args, var, ftime, date):
    #=== Config
    x_dir   = args.x_dir
    ftime   = int(ftime)
    fmt     = args.fmt
    xdata, dates = [], []
    ldate = datetime.strptime(date, fmt)
    ldate = ldate + timedelta(hours=ftime*3)
    ldate = datetime.strftime(ldate,fmt)

    xname = "%s/%s_%s" %(x_dir, var, ldate)
    xdat = np.load(xname)
    if var == "REH":
        xdat[np.where(xdat>99.9)]=99.9
        xdat = MinMaxscaler(0, 100, xdat)
    elif var == "T3H":
        xdat = xdat[44:119, 24:]
        xdat = MinMaxscaler(-50, 50, xdat)

    return(xdat)


class datasets3d(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        if x.shape[0] == y.shape[0]:
            self.rows = x.shape[0]
        else:
            print("x & y nsamples are not matched")
            sys.exit(-1)
    def __len__(self):
        return self.rows
    def __getitem__(self, idx):
        xx = torch.tensor(self.x[idx], dtype=torch.float)
        yy = torch.tensor(self.y[idx], dtype=torch.float)
        return (xx, yy)


class test_datasets3d(Dataset):
    def __init__(self, x):
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=1)
        self.x = torch.tensor(x, dtype=torch.float)
        self.rows = x.shape[0]
    def __len__(self):
        return self.rows
    def __getitem__(self, idx):
        xx = torch.tensor(self.x[idx], dtype=torch.float)
        return (xx)


def DatasetMaker(task, utc, ftime, args):

    sdate      = args.sdate
    edate      = args.edate
    var        = task
    batch_size = args.batch_size
  
    train_list, valid_list, test_list = RandomDate(sdate+utc, edate+utc)
    train_x, train_y, _               = MakeDataset(args, var, ftime, train_list)
    valid_x, valid_y, _               = MakeDataset(args, var, ftime, valid_list)
  
    train_dataset                     = datasets3d(train_x, train_y)
    train_loader                      = DataLoader(dataset = train_dataset, 
                                                   batch_size = batch_size,
                                                   shuffle = True, 
                                                   num_workers= 8,
                                                   pin_memory=True)
  
    valid_dataset                     = datasets3d(valid_x, valid_y)
    valid_loader                      = DataLoader(dataset = valid_dataset, 
                                                   batch_size = batch_size,
                                                   shuffle = False,
                                                   num_workers= 8,
                                                   pin_memory=True)
   
    return(train_loader, valid_loader) 

