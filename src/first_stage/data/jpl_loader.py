import xarray as xr
import netCDF4
import glob
import numpy as np
import os
import pickle
from datetime import datetime
from datetime import timedelta
import shutil
import sh

dictionary_path = '../data/interim/dictionaries/vars'
netcdf_path = '../data/interim/netcdf'
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
    print('pressure: ' + str(pressure))
    d1 = triplet
    triplet_delta = timedelta(hours=dt/3600)
    d0 = d1-triplet_delta
    d2 = d1+triplet_delta
    date_list = (d0, d1, d2)

    file_paths = {}
    filename = "../data/interim/experiments/" + \
        triplet.strftime("%B").lower()+"/" + str(triplet.day) + ".nc"
    ds_n = xr.open_dataset(filename)
    ds_n = ds_n.sel(pressure=pressure)
    ds_total = xr.Dataset()
    for i, date in enumerate(date_list):
        print('Downloading data for variable ' +
              var + ' for date: ' + str(date))
        directory_path = npy_stem + '/' + var.lower()
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        ds_unit = ds_n[['u', 'v', 'qv']].sel(time=date)
        T_l = 0.5*(ds_n['u'].sel(time=d0)+ds_n['u'].sel(time=d1))
        T_u = 0.5*(ds_n['u'].sel(time=d1)+ds_n['u'].sel(time=d2))
        ds_unit['umeanh'] = 0.5*(T_l+T_u)
        T_l = 0.5*(ds_n['v'].sel(time=d0)+ds_n['v'].sel(time=d1))
        T_u = 0.5*(ds_n['v'].sel(time=d1)+ds_n['v'].sel(time=d2))
        ds_unit['vmeanh'] = 0.5*(T_l+T_u)

        if not ds_total:
            ds_total = ds_unit
        else:
            ds_total = xr.concat([ds_total, ds_unit], 'time')

    if not os.path.exists(netcdf_path):
        os.makedirs(netcdf_path)
    ds_total.to_netcdf(netcdf_path+'/first_stage_raw.nc')
