#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:27:02 2019

@author: amirouyed
"""
#from pathos.multiprocessing import ProcessingPool as Pool
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

def df_loop(df,grid,dt):
    dt_inv=1/dt
    df = dfc.latlon_converter(df, grid)
    df = dfc.scaling_df_approx(df,grid,dt_inv)
    return df

def parallelize_dataframe(df, func, grid, dt, n_cores=5):
    start_time = time.time()
    print('start parallelization routine')
    df_split = np.array_split(df, n_cores)
    grid=[grid]*len(df_split)
    dt=[dt]*len(df_split)
    pool = Pool(n_cores)
    df = pd.concat(pool.starmap(func, zip(df_split,grid,dt)))
    pool.close()
    pool.join()
    print('cores: '+str(n_cores)+' seconds: ' + str(time.time() - start_time))

    return df


def df_summary(df):
    df_total=pd.DataFrame()
    for column in df:
        df_d=df[column].describe()
        df_d=df_d.to_frame()
        df_unit=df_d.T
        df_unit['quantity']=column
        #df_unit=df_unit.reset_index()
        #df_unit=df_unit.drop(columns=['index'])
        if df_total.empty:
            df_total = df_unit
        else:
            df_total = pd.concat([df_total,df_unit])
    df_total['corr_speed']= df['speed'].corr(df['speed_approx'])
    df_total['corr_u']= df['u'].corr(df['u_scaled_approx'])
    df_total['corr_v']= df['v'].corr(df['v_scaled_approx'])
    return df_total


     


def dataframe_builder(end_date, var, grid,dt):
    """build dataframe that includes data from all relevant dates"""
    dictionary_paths = glob.glob('../data/interim/dictionaries/*')
    dict_optical_paths = glob.glob(
        '../data/interim/dictionaries_optical_flow/*')
    dictionary_dict = {}
    dictionary_dict_optical = {}
    print(dictionary_paths)

    for path in dictionary_paths:
        var_name = os.path.basename(path).split('.')[0]
        dictionary_dict[var_name] = pickle.load(open(path, 'rb'))
        print(path)

    for path in dict_optical_paths:
        var_name = os.path.basename(path).split('.')[0]
        dictionary_dict_optical[var_name] = pickle.load(open(path, 'rb'))

    flow_files = dictionary_dict_optical[var]
    random_date = list(flow_files.keys())[0]   
    df = dfc.dataframe_quantum(
        flow_files[random_date], random_date, dictionary_dict)
    df= parallelize_dataframe(df, df_loop,grid, dt)
    flow_files.pop(random_date)

    for date in flow_files:
        print(date)
        file = flow_files[date]        
        df_quantum = dfc.dataframe_quantum(file, date, dictionary_dict)
        df_quantum= parallelize_dataframe(df_quantum, df_loop,grid,dt)
        df = pd.concat([df, df_quantum])

    df.set_index('datetime', inplace=True)
    df_path = '../data/interim/dataframes'
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    df.to_pickle(df_path+'/'+var+'.pkl')


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


def absolute_df(df):
    """calculates error and its absolute values"""
    df["error_u"] = df['u']-df['u_scaled_approx']
    df["error_v"] = df['v']-df['v_scaled_approx']
    df["error_u_abs"] = abs(df["error_u"])
    df["error_v_abs"] = abs(df["error_v"])
    df["u_abs"] = abs(df["u"])
    df["v_abs"] = abs(df["v"])
    df['speed'] = np.sqrt(df['u']*df['u']+df['v']*df['v'])
    df['speed_approx'] = np.sqrt(
        df['u_scaled_approx']*df['u_scaled_approx']+df['v_scaled_approx']*df['v_scaled_approx'])
    df['speed_error'] = np.sqrt(
    df['u_scaled_approx']*df['u_scaled_approx']+df['v_scaled_approx']*df['v_scaled_approx'])

    return df


def data_analysis(start_date, end_date, var, directory, cutoff):
    """perform analytics on the dataframe"""

    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.expand_frame_repr', False)
    df_path = '../data/interim/dataframes/'+var+'.pkl'
    df_path = os.path.abspath(df_path)
    df = pd.read_pickle(df_path)
    df = df[start_date:end_date]
    #import pdb; pdb.set_trace()

    df = absolute_df(df)
    if cutoff > 0:
        df = df[df.speed_error <= cutoff*df['speed'].max()]

    #import pdb; pdb.set_trace() # BREAKPOINT

    #df_printer(df, directory)

    scatter_directory = '../data/processed/scatter_'+directory
    #dfc.plotter(df[['speed', 'speed_approx']],scatter_directory, end_date)
    #            scatter_directory, end_date)
    #dfc.plotter(df[['flow_u', 'u_scaled_approx', 'u', 'error_u']],
    #            scatter_directory, end_date)
    #dfc.plotter(df[['speed', 'speed_approx', 'speed_error']],
    #            scatter_directory, end_date)

    heatmap_directory = '../data/processed/heatmaps_'+directory
    dfc.heatmap_plotter(df[['lat','lon','speed']], end_date, heatmap_directory)

    df_stats=df_summary(df)
    

    print('Done!')
    
    return df_stats
