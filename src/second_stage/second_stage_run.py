#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
,"""

import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm
from joblib import dump, load
from global_land_mask import globe
from data import extra_data_plotter as edp
from second_stage import ml_functions as mlf
import time
from second_stage import reanalysis_error as re
import xarray as xr


def ds_to_dataframe(ds, triplet_time, deltatime):
    """Get dataframe from an xarray dataset."""

    ds_unit = ds.sel(time=triplet_time)
    ds_unit['u_scaled_approx'] = 0.5*(ds['u_scaled_approx'].sel(time=triplet_time) +
                                      ds['u_scaled_approx'].sel(time=(triplet_time+deltatime)))
    ds_unit['v_scaled_approx'] = 0.5*(ds['v_scaled_approx'].sel(time=triplet_time) +
                                      ds['v_scaled_approx'].sel(time=(triplet_time+deltatime)))
    df = ds_unit.to_dataframe()
    df = df.reset_index()
    df['cos_weight'] = np.cos(df['lat']/180*np.pi)
    return df


def run(triplet_time, pressure, dt):
    """Initialize second stage of UA algorithm."""

    filename = '../data/processed/experiments/' + \
        triplet_time.strftime("%Y-%m-%d-%H:%M")+'.nc'

    ds = xr.open_dataset(filename)
    triplet_delta = datetime.timedelta(hours=dt/3600)
    df = ds_to_dataframe(ds, triplet_time, triplet_delta)
    df = df.dropna()
    df['land'] = globe.is_land(df.lat, df.lon)
    df = df.reset_index(drop=True)

    category = []
    rmse = []
    exp_list = []
    test_size = 0.95
    exp_filters = ['exp2', 'ground_t', 'df']
    print('process data...')
    dft = df.copy()
    df = re.error_calc(df, pressure, triplet_time)

    for exp_filter in exp_filters:
        print('fitting with filter ' + str(exp_filter))
        if exp_filter in ('exp2', 'error'):
            regressor, X_test0, y_test0, X_full = mlf.ml_fitter(
                df, test_size)
        elif exp_filter is 'df':
            X_test0 = df
            regressor, y_test0 = None, None
        else:
            regressor, X_test0, y_test0 = None, None, None
            print('predicting..')
        start_time = time.time()
        mlf.random_forest_calculator(df, category,  rmse,
                                     exp_filter, exp_list, regressor, X_test0, y_test0, triplet_time, X_full)
        print("--- %s seconds ---" % (time.time() - start_time))

    d = {'rmse': rmse, 'exp_filter': category}
    df_results = pd.DataFrame(data=d)
    print(df_results)

    print('done!')
