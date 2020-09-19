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


def df_builder(ds, ds_track, ds_qv_grad, date):
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

    start_time = time.time()
    df_tot = pd.merge(df, dft, on=[
        'lat', 'lon'], how='outer')
    df_tot = df_tot[df_tot.utrack_y.notna()]
    df_tot = pd.merge(df_tot, df_qv_grad.dropna(),
                      on=['lat', 'lon'], how='outer')
    dft = df_tot.drop(
        columns=['time_x', 'time_y', 'utrack_x', 'vtrack_x', 'umean_x', 'vmean_x'])
    df_tot = df_tot.drop(
        columns=['time_y', 'time_x', 'utrack_y', 'vtrack_y', 'umean_y', 'vmean_y'])

    df_tot = df_tot.rename(columns={
        'utrack_x': 'utrack', 'vtrack_x': 'vtrack', 'umean_x': 'umean', 'vmean_x': 'vmean'})
    dft = df_tot.rename(columns={
        'utrack_y': 'utrack', 'vtrack_y': 'vtrack', 'umean_y': 'umean', 'vmean_y': 'vmean'})
    dft['filter'] = 'jpl'
    df_tot = df_tot.drop_duplicates(
        ['lat', 'lon', 'filter'], keep='first')
    dft = dft.drop_duplicates(['lat', 'lon', 'filter'], keep='first')
    df_tot = df_tot.set_index(['lat', 'lon', 'filter'])
    ds = xr.Dataset.from_dataframe(df_tot)
    dft = dft.set_index(['lat', 'lon', 'filter'])
    ds_t = xr.Dataset.from_dataframe(dft)
    ds = xr.concat([ds, ds_t], 'filter')
    breakpoint()
    print("--- seconds ---" + str(time.time() - start_time))
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


def rmsvd_calculator(df, coord, rmsvd_num, rmsvd_den, filter):
    ds['cos_weight'] = np.cos(ds['lat']/180*np.pi)
    if filter is 'rean':
        erroru = ds['u_error_rean']
        errorv = ds['v_error_rean']
    else:
        erroru = ds['utrack']-df_unit.umean
        errorv = df_unit.vtrack-df_unit.vmean
    df_unit['vec_diff'] = df_unit.cos_weight * (erroru**2 + errorv**2)
    rmsvd_num = rmsvd_num + df_unit['vec_diff'].sum()
    rmsvd_den = rmsvd_den + df_unit['cos_weight'].sum()
    return rmsvd_num, rmsvd_den


def error_calc(ds):
    """Calculates and stores error of tracker algorithm into dataframe."""

    error_uj = (ds['umean'] - ds['utrack'])
    error_vj = (ds['vmean'] - ds['vtrack'])
    speed_errorj = (error_uj**2+error_vj**2)*ds['cos_weight']
    rmsvd = np.sqrt(speed_errorj.sum()/df['cos_weight'].sum())
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


def plot_preprocessor(ds, ds_track, ds_qv_grad):
    dates = ds.time.values
    coords = [(-30, 30), (-60, -30), (-90, -60), (30, 60), (60, 90)]
    # coords = [(-90, 90)]
   # filters = ['df', 'exp2', 'ground_t', 'jpl', 'rean']
    rmsvds = []
    region = []
    filter_res = []
    print('preprocessing_data...')
    ua_directory = '../data/interim/experiments/dataframes/ua/*'
    jpl_directory = '../data/interim/experiments/dataframes/jpl/*'
    files = natsorted(glob.glob(ua_directory))
    files_t = natsorted(glob.glob(jpl_directory))
    if files:
        sh.rm(files)
    if files_t:
        sh.rm(files_t)
    ds_total = xr.Dataset()
    for i, date in enumerate(dates):
        print('preprocessing: ' + str(date))
        ds_unit = df_builder(ds, ds_track[[
            'time', 'qv', 'utrack', 'vtrack', 'umean', 'vmean', 'lat', 'lon']], ds_qv_grad, date)
        ds_unit = ds_unit.assign_coords(time=str(date))
        ds_unit = ds_unit.astype(np.float32)
        if ds_total:
            ds_total = xr.concat([ds_total, ds_unit], 'time')
        else:
            ds_total = ds_unit
    breakpoint()
    for coord in tqdm(coords):
        for filter in ds_total.filter.values:
            rmsvd_num = 0
            rmsvd_den = 0
            for i, date in enumerate(dates):
                start_time = time.time()
                if filter is 'jpl':
                    df_unit = pd.read_pickle(files_t[i])
                elif filter is 'rean':
                    df_unit = pd.read_pickle(files[i])
                    df_unit = df_unit[df_unit['filter'] == 'exp2']
                else:
                    df_unit = pd.read_pickle(files[i])
                    df_unit = df_unit[df_unit['filter'] == filter]

                df_unit = df_unit.dropna()
                rmsvd_num, rmsvd_den = rmsvd_calculator(
                    df_unit, coord, rmsvd_num, rmsvd_den, filter)
                print("--- seconds ---" + str(time.time() - start_time))

            stringc = coord_to_string(coord)
            rmsvds.append(np.sqrt(rmsvd_num/rmsvd_den))
            region.append(stringc)
            filter_res.append(filter)

    d = {'latlon': region, 'exp_filter': filter_res, 'rmse': rmsvds}
    df_results = pd.DataFrame(data=d)
    return df_results


def run(triplet, pressure=500, dt=3600):
    time_string = None

    month = triplet.strftime("%B").lower()
    if dt == 3600:
        time_string = '60min'
    elif dt == 1800:
        time_string = '30min'
    else:
        raise ValueError('not supported value in dt')

    file = '../data/processed/experiments/' + \
        str(dt)+'_'+str(pressure)+'_'+month+'.nc'
    ds = xr.open_dataset(file)
    ds_track = xr.open_dataset(
        '../data/interim/experiments/'+month+'/tracked/'+time_string+'/combined/' + str(pressure)+'_'+month+'.nc')
    ds_qv_grad = xr.open_dataset(
        '../data/interim/experiments/' + month + '/tracked/' + time_string + '/combined/' + str(pressure)+'_'+month+'_'+'qv_grad.nc')
    df = plot_preprocessor(ds, ds_track, ds_qv_grad)
    df = edp.sorting_latlon(df)
    print(df)
    df.to_pickle(PATH_DF + str(dt) + '_' + month+'_' +
                 str(pressure)+'_df_results.pkl')
