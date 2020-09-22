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
from data import post_processing_calculators as ppc

PATH_DF = '../data/processed/dataframes/'
PATH_PLOT = '../data/processed/plots/'
PATH = '../data/processed/experiments/'


def plot_preprocessor(ds, ds_track, ds_qv_grad, ds_name):
    dates = ds.time.values
    print('preprocessing_data...')
    ds_total = ppc.df_merger_loop(ds, ds_track, ds_qv_grad, dates, ds_name)
    filename = PATH + ds_name+'.nc'
    ds_total = xr.open_dataset(filename)
    df_results = ppc.coord_loop(ds_total)
    return df_results, ds_total


def run(triplet, pressure=500, dt=3600):
    time_string = None

    month = triplet.strftime("%B").lower()
    if dt == 3600:
        time_string = '60min'
    elif dt == 1800:
        time_string = '30min'
    else:
        raise ValueError('not supported value in dt')

    ds_name = str(dt)+'_' + str(pressure) + '_' + \
        triplet.strftime("%B").lower() + '_merged'

    file = '../data/processed/experiments/' + \
        str(dt)+'_'+str(pressure)+'_'+month+'.nc'
    ds = xr.open_dataset(file)
    ds_track = xr.open_dataset(
        '../data/interim/experiments/'+month+'/tracked/'+time_string+'/combined/' + str(pressure)+'_'+month+'.nc')
    ds_qv_grad = xr.open_dataset(
        '../data/interim/experiments/' + month + '/tracked/' + time_string + '/combined/' + str(pressure)+'_'+month+'_'+'qv_grad.nc')
    df, ds_total = plot_preprocessor(ds, ds_track, ds_qv_grad, ds_name)
    df = edp.sorting_latlon(df)
    print(df)
    df.to_pickle(PATH_DF + str(dt) + '_' + month+'_' +
                 str(pressure)+'_df_results.pkl')
