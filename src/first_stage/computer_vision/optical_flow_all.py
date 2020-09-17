#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:15:17 2019

@author: amirouyed
"""

import os
import pickle
import cv2
import glob
import numpy as np
from tqdm import trange
from computer_vision import optical_flow_calculators as ofc
from viz import dataframe_calculators as dfc
from datetime import timedelta
import gc
import time
import xarray as xr
import datetime
import sh


def optical_flow(triplet, dt,  var, **kwargs):
    #    file_paths = pickle.load(
 #       open('../data/interim/dictionaries/vars/'+var+'.pkl', 'rb'))
    netcdf_path = '../data/interim/netcdf'

    ds = xr.open_dataset(netcdf_path+'/first_stage.nc')
    ds['flow_u'] = ds['qv'].copy()
    ds['flow_v'] = ds['qv'].copy()
    triplet_delta = datetime.timedelta(hours=dt/3600)
    start_date = triplet-triplet_delta

    frame1 = ds['qv'].sel(time=start_date).values
    frame1 = ofc.drop_nan(frame1)
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    shape = list(np.shape(frame1))
    shape.append(2)
    shape = tuple(shape)
    flow = np.zeros(shape)
    prvs = frame1
    prvs = np.nan_to_num(prvs)
    file_paths_flow = {}
    dates = []

    path = '../data/interim/flow_frames/'
    files = glob.glob(path + '*')
    if files:
        sh.rm(files)

    for date in ds.time.values[1:]:
        print(date)
        date = str(date)
        dates.append(date)
        print('flow calculation for date: ' + str(date))

        # file = file_paths[date]
        frame2 = ds['qv'].sel(time=date).values
        # frame2 = np.load(file)
        # frame2 = ds[var].sel(time=date).values

        frame2 = ofc.drop_nan(frame2)
        start_time = time.time()
        frame2 = cv2.normalize(src=frame2, dst=None, alpha=0,
                               beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        next_frame = frame2
        optical_flow = cv2.optflow.createOptFlow_DeepFlow()
        flowd = optical_flow.calc(prvs, next_frame, None)
        flow = flowd

        prvs = ofc.warp_flow(prvs, flow)
        flowd = optical_flow.calc(prvs, next_frame, None)
        flow = flow+flowd
        print("--- %s seconds ---" % (time.time() - start_time))
        print('done with deep flow')

        ds['flow_u'].loc[dict(time=date)] = flow[:, :, 0]
        ds['flow_v'].loc[dict(time=date)] = flow[:, :, 1]
        prvs = next_frame
        shape = list(np.shape(frame2))
        shape.append(2)
        shape = tuple(shape)

    filename = glob.glob(netcdf_path+'/first_stage*.nc')
    if filename:
        sh.rm(filename)
    ds.to_netcdf(netcdf_path+'/first_stage.nc')
