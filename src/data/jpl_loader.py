import xarray as xr
import netCDF4
import glob
import numpy as np
import os
import pickle
from data import make_dataset_geos5 as gd


def loader(var, pressure, start_date, end_date, dt, jpl_disk, level, triplet, sigma_random,   **kwargs):
    print('JPL loader running...')
    date = start_date
    d0 = start_date
    d1 = end_date
    date_list = gd.daterange(d0, d1, 1)
    file_paths = {}
    if var.lower() in ('utrack', 'vtrack', 'umean', 'vmean'):
        filenames = glob.glob(
            "../data/raw/jpl/processed_jpl/"+str(triplet)+"z/*")
    else:
        filenames = glob.glob("../data/raw/jpl/raw_jpl/"+str(triplet)+"z/*")
        print(filenames)
    if jpl_disk:
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
                # T = np.nan_to_num(T, nan=50)
            elif var == 'umeanh':
                print('var is umeanh')
                if date is start_date:
                    ds = xr.open_dataset(filenames[i])
                    T0 = ds.sel(pressure=pressure, method='nearest')
                    T0 = T0.get('u')
                    T0 = T0.values
                    ds = xr.open_dataset(filenames[i+1])
                    T1 = ds.sel(pressure=pressure, method='nearest')
                    T1 = T1.get('u')
                    T1 = T1.values
                    T = 0.5*(T0+T1)
                else:
                    ds = xr.open_dataset(filenames[i-1])
                    T0 = ds.sel(pressure=pressure, method='nearest')
                    T0 = T0.get('u')
                    T0 = T0.values
                    ds = xr.open_dataset(filenames[i])
                    T1 = ds.sel(pressure=pressure, method='nearest')
                    T1 = T1.get('u')
                    T1 = T1.values
                    T = 0.5*(T0+T1)
            elif var == 'vmeanh':
                if date is start_date:
                    ds = xr.open_dataset(filenames[i])
                    T0 = ds.sel(pressure=pressure, method='nearest')
                    T0 = T0.get('v')
                    T0 = T0.values
                    ds = xr.open_dataset(filenames[i+1])
                    T1 = ds.sel(pressure=pressure, method='nearest')
                    T1 = T1.get('v')
                    T1 = T1.values
                    T = 0.5*(T0+T1)
                else:
                    ds = xr.open_dataset(filenames[i-1])
                    T0 = ds.sel(pressure=pressure, method='nearest')
                    T0 = T0.get('v')
                    T0 = T0.values
                    ds = xr.open_dataset(filenames[i])
                    T1 = ds.sel(pressure=pressure, method='nearest')
                    T1 = T1.get('v')
                    T1 = T1.values
                    T = 0.5*(T0+T1)

            else:
                ds = xr.open_dataset(filenames[i])
                T = ds.sel(pressure=pressure, method='nearest')
                T = T.get(var.lower())
                T = T.values

            if var is 'qv' and sigma_random > 0:
                print('adding noise..')
               # T = np.nan_to_num(T, nan=0)
                #T = T+np.random.normal(scale=sigma_random*abs(T))

            print('shape of downloaded array: ' + str(T.shape))
            file_path = str(directory_path+'/'+str(date)+".npy")
            np.save(file_path, T)
            file_paths[date] = file_path
        dictionary_path = '../data/interim/dictionaries/vars'
        if not os.path.exists(dictionary_path):
            os.makedirs(dictionary_path)
        f = open(dictionary_path+'/' + var+'.pkl', "wb")
        pickle.dump(file_paths, f)
    else:
       # gd.downloader(start_date, end_date, var, level, False, dt)
        gd.disk_downloader(start_date, end_date, dt, level, var)
