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
import extra_data_plotter as edp


def rmsvd_calculator(df, coord, rmsvd_num, rmsvd_den):
    df_unit = df[(df.lat >= coord[0]) & (df.lat <= coord[1])]
    df_unit['cos_weight'] = np.cos(df_unit.lat/180*np.pi)
    erroru = df_unit.utrack-df_unit.umean
    errorv = df_unit.vtrack-df_unit.vmean
    df_unit['vec_diff'] = df_unit.cos_weight * (erroru**2 + errorv**2)
    rmsvd_num = rmsvd_num + df_unit['vec_diff'].sum()
    rmsvd_den = rmsvd_den + df_unit['cos_weight'].sum()
    return rmsvd_num, rmsvd_den


def df_builder(ds, ds_track, date):
    ds = ds.sel(time=date)
    ds_track = ds_track.sel(time=date)
    #ds_track = ds_track.expand_dims('filter')
    #ds_track = ds_track.assign_coords(filter=['jpl'])
    df = ds.to_dataframe().reset_index()
    dft = ds_track.to_dataframe().reset_index()
    df_tot = df.merge(dft.dropna(), on=[
        'lat', 'lon'], how='left', indicator='Exist')
    #df_dummy = df_tot[df_tot['filter'] != 'ground_t'].copy()
    df_tot = df_tot[df_tot.utrack_y.notna()]
    dft = df_tot[['utrack_y', 'vtrack_y', 'umean', 'vmean', 'lat', 'lon']]
    df_tot = df_tot.drop(
        columns=['Exist', 'time_x', 'time_y', 'utrack_y', 'umeanh', 'vmeanh'])
    df_tot = df_tot.rename(columns={'utrack_x': 'utrack'})
    df_tot = df_tot.rename(columns={'vtrack_x': 'vtrack'})
    dft = dft.rename(columns={'utrack_y': 'utrack', 'vtrack_y': 'vtrack'})
#    breakpoint()
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


def plot_preprocessor(ds, ds_track):
    dates = ds.time.values
    coords = [(-30, 30), (-60, -30), (-90, -60), (30, 60), (60, 90)]
    #coords = [(-90, 90)]
    filters = ['df', 'exp2', 'ground_t', 'jpl']
    rmsvds = []
    region = []
    filter_res = []
    print('preprocessing_data...')

    for i, date in enumerate(dates):
        print('preprocessing: ' + str(date))
        df_unit, df_t = df_builder(ds, ds_track[[
            'time', 'utrack', 'vtrack', 'umean', 'vmean', 'lat', 'lon']], date)
        df_unit.to_pickle('../data/interim/experiments/dataframes/' +
                          str(i)+'.pkl')
        df_t.to_pickle('../data/interim/experiments/dataframes/jpl/' +
                       str(i)+'.pkl')

    files = natsorted(glob.glob('../data/interim/experiments/dataframes/*'))
    files_t = natsorted(
        glob.glob('../data/interim/experiments/dataframes/jpl/*'))
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

          #  print(stringd)
           # region.append(stringd)
            stringc = coord_to_string(coord)
            rmsvds.append(np.sqrt(rmsvd_num/rmsvd_den))
            # region.append(coord)
            region.append(stringc)
            filter_res.append(filter)

    d = {'latlon': region, 'exp_filter': filter_res, 'rmse': rmsvds}
    df_results = pd.DataFrame(data=d)
    return df_results


def run():
    file = '../data/processed/experiments/july.nc'
    ds = xr.open_dataset(file)
    ds_track = xr.open_dataset(
        '../data/interim/experiments/july/tracked/60min/combined/july.nc')
    df = plot_preprocessor(ds, ds_track)
    print(df)
    df = edp.sorting_latlon(df)
    print(df)
    edp.filter_plotter(df, 'results_test', 'training data size = 5%')
