#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:27:02 2019

@author: amirouyed
"""
from multiprocessing import Pool
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from viz import dataframe_calculators as dfc
from itertools import product
import time
import gc
from tqdm import tqdm
import xarray as xr
import sh


def df_loop(df, grid, dt):
    dt_inv = 1/dt
    # df = dfc.latlon_converter(df, grid)
    df = dfc.scaling_df_approx(df, grid, dt_inv)
    return df


def parallelize_dataframe(df, grid, dt):
    start_time = time.time()
    print('start parallelization routine')

    df = df_loop(df, grid, dt)
    print("--- %s seconds ---" % (time.time() - start_time))
    return df


def df_summary(df, count):
    df_total = pd.DataFrame()
    for column in df:
        df_d = df[column].describe()
        df_d = df_d.to_frame()
        df_unit = df_d.T
        df_unit['quantity'] = column
        if df_total.empty:
            df_total = df_unit
        else:
            df_total = pd.concat([df_total, df_unit])
    df_total['corr_speed'] = df['speed'].corr(df['speed_approx'])
    df_total['corr_u'] = df['umeanh'].corr(df['u_scaled_approx'])
    df_total['corr_v'] = df['vmeanh'].corr(df['v_scaled_approx'])
    df_total['ratio_count'] = df.shape[0]/count
    df_total['count'] = count
    df_total['mean_speed_error'] = np.sqrt(df['speed_error'].mean())
    return df_total


def dataframe_builder(var, grid, dt,  **kwargs):
    netcdf_path = '../data/interim/netcdf'

    ds = xr.open_dataset(netcdf_path+'/first_stage.nc')
    df_path = '../data/interim/dataframes'
    if not os.path.exists(df_path):
        os.makedirs(df_path)

    ds_total = xr.Dataset()
    for date in ds.time.values:
        date = str(date)
        ds_unit = ds.sel(time=date)

        print('building dataframe for the date: ' + str(date))
        start_time = time.time()
        df = ds_unit.to_dataframe()
        df = df.reset_index()
        df = parallelize_dataframe(df, grid, dt)
        df = df.set_index(['lat', 'lon', 'time'])
        ds_unit = df.to_xarray()
        if not ds_total:
            ds_total = ds_unit
        else:
            ds_total = xr.concat([ds_total, ds_unit], 'time')

    filename = glob.glob(netcdf_path+'/first_stage*.nc')
    if filename:
        sh.rm(filename)
    ds_total.to_netcdf(netcdf_path+'/first_stage.nc')


def df_printer(df, directory):
    """prints dataframe summaries"""
    df_prints_path = '../data/processed/df_prints'
    if not os.path.exists(df_prints_path):
        os.makedirs(df_prints_path)
    with open(df_prints_path+'/'+directory+'.txt', 'w') as f:
        print(df[['flow_v', 'v_scaled_approx', 'v',
                  'error_v_abs']].describe(), file=f)
        print(df[['flow_u', 'u_scaled_approx', 'u',
                  'error_u_abs']].describe(), file=f)
        print(df[['v_scaled_approx', 'v', 'u_scaled_approx', 'u']].corr(
            method='pearson'), file=f)
        print(df[['speed', 'speed_error', 'speed_approx']].describe(), file=f)
        print(df[['speed', 'speed_error', 'speed_approx']].corr(
            method='pearson'), file=f)


def error_df(df):
    """calculates error and its absolute values"""
    df["error_u"] = df['umeanh']-df['u_scaled_approx']
    df["error_v"] = df['vmeanh']-df['v_scaled_approx']
    df['speed'] = np.sqrt(df['umeanh']*df['umeanh']+df['vmeanh']*df['vmeanh'])
    df['speed_approx'] = np.sqrt(
        df['u_scaled_approx']*df['u_scaled_approx']+df['v_scaled_approx']*df['v_scaled_approx'])
    df['speed_error'] = df['error_u']*df['error_u']+df['error_v']*df['error_v']

    return df


def error_ds(ds):
    ds["error_u"] = ds['umeanh']-ds['u_scaled_approx']
    ds["error_v"] = ds['vmeanh']-ds['v_scaled_approx']
    ds['speed'] = np.sqrt(ds['umeanh']*ds['umeanh']+ds['vmeanh']*ds['vmeanh'])
    ds['speed_approx'] = np.sqrt(
        ds['u_scaled_approx']*ds['u_scaled_approx']+ds['v_scaled_approx']*ds['v_scaled_approx'])
    ds['speed_error'] = ds['error_u']*ds['error_u']+ds['error_v']*ds['error_v']

    return ds


def df_concatenator(dataframes_dict):
    df = pd.DataFrame()
    print('concatenating dataframes for all dates for further analysis:')
    for date in tqdm(dataframes_dict):
        # if date >= start_date and date <= end_date:
        gc.collect()
        df_path = dataframes_dict[date]
        df_unit = pd.read_pickle(df_path)
        df_unit = df_unit.reset_index(drop=True)

        if df.empty:
            df = df_unit
        else:
            df = df + df_unit

    df = df/2
    df = error_df(df)
    df = df[['lon', 'lat', 'speed',  'qv', 'speed_approx', 'speed_error',
             'error_v', 'error_u', 'u_scaled_approx', 'v_scaled_approx', 'vmeanh', 'umeanh']]

    df['cos_weight'] = np.cos(df['lat']/180*np.pi)
    return df


def df_to_netcdf(dataframes_dict, triplet):
    ds = xr.Dataset()

    for date in dataframes_dict:
        df_path = dataframes_dict[date]
        df = pd.read_pickle(df_path)
        lon = np.arange(df['lon'].min(), df['lon'].max() + 0.0625, 0.0625)
        lat = np.arange(df['lat'].min(), df['lat'].max() + 0.0625, 0.0625)
        columns = np.array(df.columns.values.tolist())
        columns_diff = np.array(['y', 'x', 'flow_u', 'flow_v', 'lat', 'lon'])
        columns = np.setdiff1d(columns, columns_diff)

        for j, var in enumerate(columns):
            da = xr.DataArray()
            varval = df.pivot('lat', 'lon', var).values

            da = xr.DataArray(varval, coords=[
                lat, lon], dims=['lat', 'lon'])
            da = da.expand_dims('time')
            da = da.assign_coords(time=[date])

            if(j == 0):
                ds_unit = xr.Dataset({var: da})
            else:
                ds_unit_m = xr.Dataset({var: da})
                ds_unit = xr.merge([ds_unit, ds_unit_m])

        if not ds:
            ds = ds_unit
        else:
            ds = xr.concat([ds, ds_unit], 'time')

    ds.to_netcdf('../data/interim/experiments/first_stage_amvs/' +
                 triplet.strftime("%Y-%m-%d-%H:%M")+'.nc')


def data_analysis(triplet,  **kwargs):
    """perform analytics on the dataframe"""

    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.expand_frame_repr', False)
    netcdf_path = '../data/interim/netcdf'

    ds = xr.open_dataset(netcdf_path+'/first_stage.nc')

    ds_final = ds.sel(time=str(ds.time.values[0]))
    dl = ds['u_scaled_approx'].sel(time=str(ds.time.values[0]))
    du = ds['u_scaled_approx'].sel(time=str(ds.time.values[1]))
    ds_final['u_scaled_approx'] = 0.5*(dl+du)

    dl = ds['v_scaled_approx'].sel(time=str(ds.time.values[0]))
    du = ds['v_scaled_approx'].sel(time=str(ds.time.values[1]))
    ds_final['v_scaled_approx'] = 0.5*(dl+du)
    ds_final = error_ds(ds_final)
    ds_final.to_netcdf('../data/interim/experiments/first_stage_amvs/' +
                       triplet.strftime("%Y-%m-%d-%H:%M")+'.nc')
    df = ds_final.to_dataframe()
    df = df.reset_index()
    count = df.shape[0]
    df = df.dropna()
    df_stats = df_summary(df, count)
    print('Done!')

    return df_stats
