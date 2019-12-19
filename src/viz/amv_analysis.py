#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:27:02 2019

@author: amirouyed
"""
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
import parmap
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from viz import dataframe_calculators as dfc
from itertools import product
import time
from tqdm import tqdm


def df_loop(df, grid, dt):
    dt_inv = 1/dt
    df = dfc.latlon_converter(df, grid)
    df = dfc.scaling_df_approx(df, grid, dt_inv)
    return df


def parallelize_dataframe(df, func, grid, dt, n_cores=5):
    start_time = time.time()
    print('start parallelization routine')
    df_split = np.array_split(df, n_cores)
    grid = [grid]*len(df_split)
    dt = [dt]*len(df_split)
    pool = Pool(n_cores)
    df = pd.concat(pool.starmap(func, zip(df_split, grid, dt)))
    pool.close()
    pool.join()
    print('cores: '+str(n_cores)+' seconds: ' + str(time.time() - start_time))

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
    #df_total['corr_u']= df['u'].corr(df['u_scaled_approx'])
    #df_total['corr_v']= df['v'].corr(df['v_scaled_approx'])
    # df_total['initial_count']=count
    df_total['ratio_count'] = df.shape[0]/count
    df_total['mean_speed_error'] = df['speed_error'].mean()
    return df_total


def dataframe_builder(end_date, var, grid, dt, **kwargs):
    """build dataframe that includes data from all relevant dates"""
    dictionary_paths = glob.glob('../data/interim/dictionaries/vars/*')
    dictionary_path = '../data/interim/dictionaries/'

    dict_optical_paths = glob.glob(
        '../data/interim/dictionaries_optical_flow/*')
    dictionary_dict = {}
    dictionary_dict_optical = {}
    dictionary_dataframes = {}

    for path in dictionary_paths:
        var_name = os.path.basename(path).split('.')[0]
        dictionary_dict[var_name] = pickle.load(open(path, 'rb'))

    for path in dict_optical_paths:
        var_name = os.path.basename(path).split('.')[0]
        dictionary_dict_optical[var_name] = pickle.load(open(path, 'rb'))

    df_path = '../data/interim/dataframes'
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    flow_files = dictionary_dict_optical[var]

    for date in flow_files:
        print('building dataframe for the date: ' + str(date))
        file = flow_files[date]
        df = dfc.dataframe_quantum(file, date, dictionary_dict)
        df = parallelize_dataframe(df, df_loop, grid, dt)
        df.set_index('datetime', inplace=True)
        path = df_path+'/'+var+'_'+str(date)+'.pkl'
        df.to_pickle(path)
        dictionary_dataframes[date] = path
        f = open(dictionary_path+'/dataframes.pkl', "wb")
        pickle.dump(dictionary_dataframes, f)


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
    df["error_u"] = df['u']-df['u_scaled_approx']
    df["error_v"] = df['v']-df['v_scaled_approx']
    df['speed'] = np.sqrt(df['u']*df['u']+df['v']*df['v'])
    df['speed_approx'] = np.sqrt(
        df['u_scaled_approx']*df['u_scaled_approx']+df['v_scaled_approx']*df['v_scaled_approx'])
    df['speed_error'] = np.sqrt(
        df['error_u']*df['error_u']+df['error_v']*df['error_v'])

    return df


def df_concatenator(dataframes_dict, start_date, end_date):
    df = pd.DataFrame()
    print('concatenating dataframes for all dates for further analysis:')
    for date in tqdm(dataframes_dict):
        if date >= start_date and date <= end_date:
            df_path = dataframes_dict[date]
            df_unit = pd.read_pickle(df_path)
            df_unit = df_unit[['lon', 'lat', 'u', 'v',
                               'u_scaled_approx', 'v_scaled_approx', 'qvdens']]
            df_unit = error_df(df_unit)
            df_unit = df_unit[['speed', 'speed_approx', 'speed_error']]

            df_unit = df_unit.apply(pd.to_numeric, downcast='float')
            if df.empty:
                df = df_unit
            else:
                df = pd.concat([df, df_unit])
    return df


def data_analysis(start_date, end_date, var, path, cutoff, **kwargs):
    """perform analytics on the dataframe"""

    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.expand_frame_repr', False)
    dict_path = '../data/interim/dictionaries/dataframes.pkl'
    dataframes_dict = pickle.load(open(dict_path, 'rb'))
    df = df_concatenator(dataframes_dict, start_date, end_date)
    count = df.shape[0]
    if cutoff > 0:
        df = df[df.speed_error <= cutoff]

    #df_printer(df, directory)

    #scatter_directory = '../data/processed/scatter_'+directory
    #dfc.plotter(df[['speed', 'speed_approx']],scatter_directory, end_date)
    #            scatter_directory, end_date)
    # dfc.plotter(df[['flow_u', 'u_scaled_approx', 'u', 'error_u']],
    #            scatter_directory, end_date)
    # dfc.plotter(df[['speed', 'speed_approx', 'speed_error']],
    #            scatter_directory, end_date)

    heatmap_directory = '../data/processed/heatmaps/'+path
    print(heatmap_directory)

    #dfc.heatmap_plotter(df[['lat','lon','speed_approx','speed','qvdens']], end_date, heatmap_directory)

    df_stats = df_summary(df, count)

    print('Done!')

    return df_stats
