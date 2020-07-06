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
from sklearn.utils import resample
PATH_DF = '../data/processed/dataframes/'
PATH_PLOT = '../data/processed/plots/'


def rmsvd_calculator(df, coord, rmsvd_num, rmsvd_den):
    df_unit = df[(df.lat >= coord[0]) & (df.lat <= coord[1])]
    df_unit['cos_weight'] = np.cos(df_unit.lat/180*np.pi)
    erroru = df_unit.utrack-df_unit.umean
    errorv = df_unit.vtrack-df_unit.vmean
    df_unit['vec_diff'] = df_unit.cos_weight * (erroru**2 + errorv**2)
    rmsvd_num = rmsvd_num + df_unit['vec_diff'].sum()
    rmsvd_den = rmsvd_den + df_unit['cos_weight'].sum()
    return rmsvd_num, rmsvd_den


def df_builder(ds, ds_track, ds_qv_grad, date):
    ds = ds.sel(time=date)
    ds = ds.astype(np.float32)
    ds_track = ds_track.sel(time=date)
    ds_track = ds_track.astype(np.float32)
    ds_qv_grad = ds_qv_grad.sel(time=date)
    ds_qv_grad = ds_qv_grad.astype(np.float32)
    df = ds.to_dataframe().reset_index()
    dft = ds_track.to_dataframe().reset_index()
    df_qv_grad = ds_qv_grad.to_dataframe().reset_index()
    df_qv_grad = df_qv_grad.drop(columns=['time'])
    df_tot = df.merge(dft.dropna(), on=[
        'lat', 'lon'], how='left')
    df_tot = df_tot[df_tot.utrack_y.notna()]
    df_tot = df_tot.merge(df_qv_grad.dropna(), on=['lat', 'lon'], how='left')
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
    return df_tot, dft


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


def rmsvd_calculator(df, coord, rmsvd_num, rmsvd_den):
    df_unit = df[(df.lat >= coord[0]) & (df.lat <= coord[1])]
    df_unit['cos_weight'] = np.cos(df_unit.lat/180*np.pi)
    erroru = df_unit.utrack-df_unit.umean
    errorv = df_unit.vtrack-df_unit.vmean
    df_unit['vec_diff'] = df_unit.cos_weight * (erroru**2 + errorv**2)
    rmsvd_num = rmsvd_num + df_unit['vec_diff'].sum()
    rmsvd_den = rmsvd_den + df_unit['cos_weight'].sum()
    return rmsvd_num, rmsvd_den


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
    filters = ['df', 'exp2', 'ground_t', 'jpl']
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
    for i, date in enumerate(dates):
        print('preprocessing: ' + str(date))
        df_unit, df_t = df_builder(ds, ds_track[[
            'time', 'qv', 'utrack', 'vtrack', 'umean', 'vmean', 'lat', 'lon']], ds_qv_grad, date)
        df_t['filter'] = 'jpl'
        df_unit.to_pickle('../data/interim/experiments/dataframes/ua/' +
                          str(i)+'.pkl')
        df_t.to_pickle('../data/interim/experiments/dataframes/jpl/' +
                       str(i)+'.pkl')

    files = natsorted(glob.glob(ua_directory))
    files_t = natsorted(glob.glob(jpl_directory))
    for coord in tqdm(coords):
        for filter in filters:
            rmsvd_num = 0
            rmsvd_den = 0
            for i, date in enumerate(dates):
                if (filter is 'jpl'):
                    df_unit = pd.read_pickle(files_t[i])
                else:
                    df_unit = pd.read_pickle(files[i])
                    df_unit = df_unit[df_unit['filter'] == filter]
                df_unit = df_unit.dropna()

                rmsvd_num, rmsvd_den = rmsvd_calculator(
                    df_unit, coord, rmsvd_num, rmsvd_den)

            stringc = coord_to_string(coord)
            rmsvds.append(np.sqrt(rmsvd_num/rmsvd_den))
            region.append(stringc)
            filter_res.append(filter)

    d = {'latlon': region, 'exp_filter': filter_res, 'rmse': rmsvds}
    df_results = pd.DataFrame(data=d)
    return df_results


def run(pressure):
    file = '../data/processed/experiments/'+str(pressure)+'_july.nc'
    ds = xr.open_dataset(file)
    ds_track = xr.open_dataset(
        '../data/interim/experiments/july/tracked/60min/combined/'+str(pressure)+'_july.nc')
    ds_qv_grad = xr.open_dataset(
        '../data/interim/experiments/july/tracked/60min/combined/'+str(pressure)+'_july_qv_grad.nc')
    df = plot_preprocessor(ds, ds_track, ds_qv_grad)
    df = edp.sorting_latlon(df)
    print(df)
    df.to_pickle(PATH_DF+str(pressure)+'_df_results.pkl')
    edp.filter_plotter(df, PATH_PLOT+str(pressure) +
                       '_results_test', 'training data size = 5%')
