import xarray as xr
import netCDF4
import glob
import numpy as np
import os
import pickle
from datetime import datetime
from datetime import timedelta


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


def loader(var, pressure, start_date, end_date, dt, jpl_disk, level, triplet, sigma_random,   **kwargs):
    print('JPL loader running...')
    d1 = start_date
    triplet_delta = timedelta(hours=1)
    d0 = d1-triplet_delta
    d2 = d1+triplet_delta

    date_list = (d0, d1, d2)
    file_paths = {}
    filename = "../data/interim/experiments/july/01.nc"
    ds = xr.open_dataset(filename)
    ds = ds.sel(pressure=850)

    for date in date_list:
        print('Downloading data for variable ' +
              var + ' for date: ' + str(date))

        if var.lower() in ('umeanh'):
            T = ds['u'].sel(time=d1)
            T = T.values
            T = np.squeeze(T)

        elif var.lower() in ('vmeanh'):
            T = ds['v'].sel(time=d1)
            T = T.values
            T = np.squeeze(T)
        else:

            T = ds[var.lower()].sel(time=date)
            T = T.values
            T = np.squeeze(T)

        directory_path = '../data/raw/'+var.lower()
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        print('shape of downloaded array: ' + str(T.shape))
        file_path = str(directory_path+'/'+str(date)+".npy")
        np.save(file_path, T)
        file_paths[date] = file_path
    dictionary_path = '../data/interim/dictionaries/vars'
    if not os.path.exists(dictionary_path):
        os.makedirs(dictionary_path)
    f = open(dictionary_path+'/' + var+'.pkl', "wb")
    pickle.dump(file_paths, f)
