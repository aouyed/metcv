import xarray as xr
import netCDF4
import glob
import numpy as np
import os
import pickle 
from data import make_dataset_geos5 as gd
def loader(var, pressure, start_date, end_date, dt,  **kwargs):
    date=start_date
    d0=start_date
    d1=end_date
    date_list = gd.daterange(d0, d1, 1)
    file_paths = {}
    if var.lower() == 'utrack' or var.lower() =='vtrack':
        filenames = glob.glob("../data/raw/jpl/processed_jpl/*")
    else:
        filenames = glob.glob("../data/raw/jpl/raw_jpl/*")

    
    for i, date in enumerate(date_list):
        print('Downloading data for variable ' + var + ' for date: ' + str(date))
        directory_path = '../data/raw/'+var.lower()
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

 
        if var.lower() == 'utrack' or var.lower() =='vtrack':
            ds = xr.open_dataset(filenames[0])
            T=ds.get(var.lower())
        else:
            ds = xr.open_dataset(filenames[i])
            T = ds.sel(pressure=pressure, method='nearest')
            T=T.get(var.lower())
        T=T.values
        T=np.nan_to_num(T)
        print('shape of downloaded array: ' + str(T.shape))
        file_path = str(directory_path+'/'+str(date)+".npy")
        np.save(file_path, T)
        file_paths[date] = file_path
    dictionary_path = '../data/interim/dictionaries/vars'
    if not os.path.exists(dictionary_path):
        os.makedirs(dictionary_path)
    f = open(dictionary_path+'/' + var+'.pkl', "wb")
    pickle.dump(file_paths, f)
