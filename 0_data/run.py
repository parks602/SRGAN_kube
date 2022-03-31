#0. dataset

import sys
import os
import warnings
import argparse

import numpy as np

from datetime import datetime, timedelta

def Data_preprocess(args):
    #=== Config
    fmt         = args.fmt
    load_x_dir  = args.NWPD_dir
    load_y_dir  = args.ldaps_dir
    save_x_dir  = args.x_dir
    save_y_dir  = args.y_dir
    tasks       = args.tasks
    sdate       = args.sdate
    edate       = args.edate
    ftimes      = args.ftimes
    utcs        = args.utcs

    for utc in utcs:
        dt_sdate    = datetime.strptime(sdate+utc, fmt)  ### str -> datetime
        dt_edate    = datetime.strptime(edate+utc, fmt)
        now         = dt_sdate
        day_list    = []
    
        while now<=dt_edate:
            ex_sdate = now.strftime(fmt)
            day_list.append(ex_sdate)
            now = now + timedelta(days=1)
    
        print('Data date collect finish %s ~ %s'%(dt_sdate,dt_edate))
        print('Data len : ' , len(day_list))
    
        for date in day_list:
            print('%s data preprocess is started'%(date))
            xname  = "%s/umgl_n128.%s.npz" %(load_x_dir, date)
            dt_date    = datetime.strptime(date, fmt)
            for task in tasks:
                for ftime in ftimes:
                    ftime_date = datetime.strftime(dt_date+timedelta(hours=ftime * 3),fmt)
                    yname  = "%s/REA_VSRT_GRD_LDPS_BARN_%s.%s.npy" %(load_y_dir%(task), task, ftime_date)
                if not FileExists(xname) or not FileExists(yname):
                    print(task, ftime_date, 'is not exist')
                else:
                    xdata = np.load(xname)
                    ydata = np.load(yname)
                    if task == 'T3H':
                        xdata = xdata['t'][ftime,:,:]-273.15
                        xdata = xdata.swapaxes(0,1)
                        xdata = xdata[44:119, 24:]
                    elif task == 'REH':
                        xdata = xdata['r'][ftime,:,:]
                        xdata = xdata.swapaxes(0,1)
                        xdata = xdata[44:119, 24:]
                    np.save('%s/%s_%s.npy'%(save_x_dir, task, ftime_date), xdata)
                    np.save('%s/%s_%s.npy'%(save_y_dir, task, ftime_date), ydata)


def FileExists(path):
    if not os.path.exists(path):
        print("Can't Find : %s" %(path))
        return False
    else:
        return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fmt', type=str, default = '%Y%m%d%H')
    parser.add_argument('--NWPD_dir', type=str, default = '/home/pkw/2022/DATA/NWPD')
    parser.add_argument('--ldaps_dir', type=str, default = '/home/pkw/2022/DATA/LDAPS/%s/')
    parser.add_argument('--x_dir', type=str, default = '/home/pkw/2022/DATA/data/x')
    parser.add_argument('--y_dir', type=str, default = '/home/pkw/2022/DATA/data/y')
    parser.add_argument('--tasks', type=list, default = ['T3H', 'REH'])
    parser.add_argument('--utcs',  type=list, default = ['00', '12'])
    parser.add_argument('--ftimes', type=list, default = [2,3])
    parser.add_argument('--sdate', type=str, default = '20210301')
    parser.add_argument('--edate', type=str, default = '20210330')

    args = parser.parse_args()
    Data_preprocess(args)

