import time
import pdb
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import numpy as np
from tqdm import tqdm
import pickle
import glob
from natsort import natsorted
import pandas as pd
from data import extra_data_plotter as edp
import sh
import dask
import dask.dataframe as dd
from sklearn.utils import resample
PATH_DF = '../data/processed/dataframes/'
PATH_PLOT = '../data/processed/plots/'
PATH = '../data/processed/experiments/'


def df_builder(ds, ds_track, ds_qv_grad, date, ds_name):
    date = str(date)
    print('string date: ' + date)
    ds = ds.sel(time=date)
    ds = ds.astype(np.float32)
    ds_track = ds_track.sel(time=date)
    ds_track = ds_track.astype(np.float32)
    ds_qv_grad = ds_qv_grad.sel(time=date)
    ds_qv_grad = ds_qv_grad.astype(np.float32)
    df = ds.to_dataframe().reset_index()
    dft = ds_track.to_dataframe().reset_index()
    dft = dft.dropna()
    df_qv_grad = ds_qv_grad.to_dataframe().reset_index()
    df_qv_grad = df_qv_grad.drop(columns=['time'])

    #start_time = time.time()

    df_tot = pd.merge(df, dft, on=[
        'lat', 'lon'], how='outer')
    df_tot = df_tot[df_tot.utrack_y.notna()]
    df_tot = pd.merge(df_tot, df_qv_grad.dropna(),
                      on=['lat', 'lon'], how='outer')
    dft = df_tot.drop(
        columns=['time_x', 'time_y', 'utrack_x', 'vtrack_x', 'umean_x', 'vmean_x'])
    df_tot = df_tot.drop(
        columns=['time_x', 'time_y', 'utrack_y', 'vtrack_y', 'umean_y', 'vmean_y'])

    df_tot = df_tot.rename(columns={
                           'utrack_x': 'utrack', 'vtrack_x': 'vtrack', 'umean_x': 'umean', 'vmean_x': 'vmean'})
    dft = dft.rename(columns={
        'utrack_y': 'utrack', 'vtrack_y': 'vtrack', 'umean_y': 'umean', 'vmean_y': 'vmean'})
    dft['filter'] = 'jpl'
    df_tot = df_tot.drop_duplicates(['lat', 'lon', 'filter'], keep='first')
    dft = dft.drop_duplicates(['lat', 'lon', 'filter'], keep='first')
    #print("--- seconds ---" + str(time.time() - start_time))
    dft['cos_weight'] = np.cos(dft['lat']/180*np.pi)
    df_tot['cos_weight'] = np.cos(df_tot['lat']/180*np.pi)
    dft = dft.drop_duplicates(['lat', 'lon', 'filter'], keep='first')
    df_tot = df_tot.set_index(['lat', 'lon', 'filter'])
    ds = xr.Dataset.from_dataframe(df_tot)
    dft = dft.set_index(['lat', 'lon', 'filter'])
    ds_t = xr.Dataset.from_dataframe(dft)
    ds = xr.concat([ds, ds_t], 'filter')
    print(ds)
    return ds


def coord_to_string(coord):
    if coord[0] < 0:
        lowlat = str(abs(coord[0])) + '°S'
    else:
        lowlat = str(coord[0]) + '°N'

    if coord[1] < 0:
        uplat = str(abs(coord[1])) + '°S'
    else:
        uplat = str(coord[1]) + '°N'
    stringd = str(str(lowlat)+',' + str(uplat))
    return stringd


def rmsvd_calculator(ds, filter):
    df_unit = ds.to_dataframe()
    df_unit = df_unit.reset_index().dropna()

    if filter == 'rean':
        erroru = df_unit.u_error_rean
        errorv = df_unit.v_error_rean
    else:
        erroru = df_unit.utrack-df_unit.umean
        errorv = df_unit.vtrack-df_unit.vmean
    df_unit['vec_diff'] = df_unit.cos_weight * (erroru**2 + errorv**2)
    rmsvd_num = df_unit['vec_diff'].sum()
    rmsvd_den = df_unit['cos_weight'].sum()
    rmsvd = np.sqrt(rmsvd_num/rmsvd_den)
    return rmsvd


def coord_to_string(coord):
    if coord[0] < 0:
        lowlat = str(abs(coord[0])) + '°S'
    else:
        lowlat = str(coord[0]) + '°N'

    if coord[1] < 0:
        uplat = str(abs(coord[1])) + '°S'
    else:
        uplat = str(coord[1]) + '°N'
    stringd = str(str(lowlat)+',' + str(uplat))
    return stringd


def df_merger_loop(ds, ds_track, ds_qv_grad, dates, ds_name):
    ds_total = xr.Dataset()
    for i, date in enumerate(dates):
        print('preprocessing: ' + str(date))
        start_time = time.time()
        ds_unit = df_builder(ds, ds_track[[
            'time', 'qv', 'utrack', 'vtrack', 'umean', 'vmean', 'lat', 'lon']], ds_qv_grad, date, ds_name)
        ds_unit = ds_unit.assign_coords(time=str(date))
        ds_unit = ds_unit.astype(np.float32)
        if ds_total:
            ds_total = xr.concat([ds_total, ds_unit], 'time')
        else:
            ds_total = ds_unit
        print("--- seconds ---" + str(time.time() - start_time))

    filename = PATH + ds_name+'.nc'
    files = glob.glob(filename+'*')
    if files:
        sh.rm(files)
    ds_total.to_netcdf(filename,  mode='w')

    return ds_total


def coord_loop(ds_total):
    coords = [(-30, 30), (-60, -30), (-90, -60), (30, 60), (60, 90)]
    rmsvds = []
    region = []
    filter_res = []
    filters = ds_total.filter.values
    filters = np.append(filters, 'rean')
    for coord in tqdm(coords):
        start_time = time.time()
        ds_coord = ds_total.sel(lat=slice(coord[0], coord[1]))
        for filter_u in filters:
            print(filter_u)
            if filter_u != 'rean':
                ds_filter = ds_coord.sel(filter=filter_u)
            else:
                ds_filter = ds_coord.sel(filter='df')

            rmsvd = rmsvd_calculator(ds_filter, filter_u)
            stringc = coord_to_string(coord)
            rmsvds.append(rmsvd)
            region.append(stringc)
            filter_res.append(filter_u)
        print("--- seconds ---" + str(time.time() - start_time))
    d = {'latlon': region, 'exp_filter': filter_res, 'rmse': rmsvds}
    df_results = pd.DataFrame(data=d)
    return df_results
