import xarray as xr
import netCDF4
import glob
import numpy as np
import os
import pickle
from datetime import datetime
from datetime import timedelta
from natsort import natsorted


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
    print('var')
    print(var)
    filenames = natsorted(glob.glob(
        "../data/raw/experiments/07_01_2006/"+var+"/850/"+str(triplet)+"zh/*"))
    if var is 'umeanh':
        filenames = natsorted(glob.glob(
            "../data/raw/experiments/07_01_2006/u/850/"+str(triplet)+"zh/*"))
    elif var is 'vmeanh':
        filenames = natsorted(glob.glob(
            "../data/raw/experiments/07_01_2006/v/850/"+str(triplet)+"zh/*"))

    print(filenames)
    for i, date in enumerate(date_list):
        print('Downloading data for variable ' +
              var + ' for date: ' + str(date))
        directory_path = '../data/interim/'+var.lower()
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        if var is 'umeanh':
            T = np.load(filenames[1])

        elif var is 'vmeanh':
            T = np.load(filenames[1])

        else:
            T = np.load(filenames[i])
        T = np.squeeze(T)
        T = np.float32(T)
        print('shape of downloaded array: ' + str(T.shape))
        file_path = str(directory_path+'/'+str(date)+".npy")
        np.save(file_path, T)
        file_paths[date] = file_path
    dictionary_path = '../data/interim/dictionaries/vars'
    if not os.path.exists(dictionary_path):
        os.makedirs(dictionary_path)
    f = open(dictionary_path+'/' + var+'.pkl', "wb")
    pickle.dump(file_paths, f)
