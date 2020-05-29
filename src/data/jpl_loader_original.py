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
    date = start_date
    d0 = start_date
    d1 = end_date
    date_list = daterange(d0, d1, 1)
    file_paths = {}
    if var.lower() in ('utrack', 'vtrack', 'umean', 'vmean'):
        filenames = glob.glob(
            "../data/raw/jpl/processed_jpl/"+str(triplet)+"z/*")
    else:
        filenames = glob.glob("../data/raw/jpl/raw_jpl/"+str(triplet)+"z/*")
        print(filenames)
    for i, date in enumerate(date_list):
        print('Downloading data for variable ' +
              var + ' for date: ' + str(date))
        directory_path = '../data/raw/'+var.lower()
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if var.lower() in ('utrack', 'vtrack', 'umean', 'vmean'):
            ds = xr.open_dataset(filenames[0])
            T = ds.get(var.lower())
            T = T.values
        elif var == 'umeanh':
            ds = xr.open_dataset(filenames[1])
            T = ds.sel(pressure=pressure, method='nearest')
            T = T.get('u')
            T = T.values

        elif var == 'vmeanh':
            ds = xr.open_dataset(filenames[1])
            T = ds.sel(pressure=pressure, method='nearest')
            T = T.get('v')
            T = T.values

        else:
            ds = xr.open_dataset(filenames[i])
            T = ds.sel(pressure=pressure, method='nearest')
            T = T.get(var.lower())
            T = T.values

        print('shape of downloaded array: ' + str(T.shape))
        file_path = str(directory_path+'/'+str(date)+".npy")
        np.save(file_path, T)
        file_paths[date] = file_path
    dictionary_path = '../data/interim/dictionaries/vars'
    if not os.path.exists(dictionary_path):
        os.makedirs(dictionary_path)
    f = open(dictionary_path+'/' + var+'.pkl', "wb")
    pickle.dump(file_paths, f)
