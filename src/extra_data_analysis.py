#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
,"""

import pdb
from viz import amv_analysis as aa
from viz import dataframe_calculators as dfc
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm
from joblib import dump, load
from global_land_mask import globe
import reverse_geocoder
import extra_data_plotter as edp
import ml_functions as mlf
import time
import reanalysis_error as re
import xarray as xr

R = 6373.0


def ds_to_dataframe(ds, triplet_time, deltatime):
    ds_unit = ds.sel(time=triplet_time)
    ds_unit['u_scaled_approx'] = 0.5*(ds['u_scaled_approx'].sel(time=triplet_time) +
                                      ds['u_scaled_approx'].sel(time=(triplet_time+deltatime)))
    ds_unit['v_scaled_approx'] = 0.5*(ds['v_scaled_approx'].sel(time=triplet_time) +
                                      ds['v_scaled_approx'].sel(time=(triplet_time+deltatime)))
    df = ds_unit.to_dataframe()
    df = df.reset_index()
    df['cos_weight'] = np.cos(df['lat']/180*np.pi)
    return df


def run(triplet_time):

    triplet_time = datetime.datetime(2006, 7, 1, 0, 0, 0, 0)

    filename = '../data/processed/experiments/' + \
        triplet_time.strftime("%Y-%m-%d-%H:%M")+'.nc'

    ds = xr.open_dataset(filename)
    triplet_delta = datetime.timedelta(hours=1)
    df = ds_to_dataframe(ds, triplet_time, triplet_delta)

    print(df.shape)
    df = df.dropna()
    print('non nan shape')
    print(df.shape)
    df['land'] = globe.is_land(df.lat, df.lon)
    df = df.reset_index(drop=True)

    category = []
    rmse = []
    latlon = []
    test_sizes = []
    exp_list = []
    only_land = False

    # latdowns = [-30, 30, 60, -60, -90]
    # latups = [30, 60, 90, -30, -60]
    latdowns = [-90]
    latups = [90]
    test_size = 0.95
    exp_filters = ['exp2', 'ground_t', 'df']
    print('process data...')
    dft = df.copy()
    df = re.error_calc(df)

    for exp_filter in exp_filters:
        print('fitting with filter ' + str(exp_filter))
        if exp_filter in ('exp2', 'error'):
            regressor, X_test0, y_test0 = mlf.ml_fitter('uv', df,
                                                        'rf', rmse, test_size, only_land, -90, 90, exp_filter)
        else:
            regressor, X_test0, y_test0 = 0, 0, 0
            print('predicting..')
        for i, latdown in enumerate(tqdm(latdowns)):
            start_time = time.time()
            mlf.latitude_selector(df, latdown, latups[i], category,  rmse, latlon,  test_size,
                                  test_sizes, only_land, exp_filter, exp_list, regressor, X_test0, y_test0, triplet_time)
            print("--- %s seconds ---" % (time.time() - start_time))

    d = {'latlon': latlon, 'categories': category,
         'rmse': rmse, 'exp_filter': exp_list}

    df_results = pd.DataFrame(data=d)

    df_results.to_pickle("df_results.pkl")

    print('done!')
    print(df_results)
