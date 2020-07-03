import xarray as xr
import netCDF4
import glob
import numpy as np
import os
import pickle
from datetime import datetime
from datetime import timedelta
import shutil

dictionary_path = '../data/interim/dictionaries/vars'
npy_stem = '../data/interim/npys'


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


def resetter():
    if os.path.exists(dictionary_path):
        shutil.rmtree(dictionary_path, ignore_errors=False, onerror=None)
    if os.path.exists(npy_stem):
        shutil.rmtree(npy_stem, ignore_errors=False, onerror=None)


def loader(var, pressure,  dt,  triplet,   **kwargs):
    print('JPL loader running...')

    d1 = triplet
    triplet_delta = timedelta(hours=dt/3600)
    d0 = d1-triplet_delta
    d2 = d1+triplet_delta
    date_list = (d0, d1, d2)

    file_paths = {}
    filename = "../data/interim/experiments/july/" + str(triplet.day) + ".nc"
    ds_n = xr.open_dataset(filename)
    ds_n = ds_n.sel(pressure=pressure)

    for i, date in enumerate(date_list):
        print('Downloading data for variable ' +
              var + ' for date: ' + str(date))
        directory_path = npy_stem + '/' + var.lower()
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if var == 'umeanh':
            T = ds_n['u'].sel(time=d1)
            T = T.values
            T = np.squeeze(T)
        elif var == 'vmeanh':
            T = ds_n['v'].sel(time=d1)
            T = T.values
            T = np.squeeze(T)

        else:
            T = ds_n[var.lower()].sel(time=date_list[i])
            T = T.values
            T = np.squeeze(T)

        print('shape of downloaded array: ' + str(T.shape))
        file_path = str(directory_path+'/'+str(date)+".npy")
        np.save(file_path, T)
        file_paths[date] = file_path
    if not os.path.exists(dictionary_path):
        os.makedirs(dictionary_path)
    f = open(dictionary_path+'/' + var+'.pkl', "wb")
    pickle.dump(file_paths, f)
