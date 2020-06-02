#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:15:17 2019

@author: amirouyed
"""

import os
import pickle
import cv2
from skimage import util
from computer_vision import cross_correlation as cc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from skimage.feature import register_translation
from tqdm import trange
from computer_vision import optical_flow_calculators as ofc
from viz import dataframe_calculators as dfc
from datetime import timedelta
import gc
import time
import xarray as xr
import datetime


def optical_flow(triplet, dt,  var, **kwargs):
    file_paths = pickle.load(
        open('../data/interim/dictionaries/vars/'+var+'.pkl', 'rb'))

    triplet_delta = datetime.timedelta(hours=dt/3600)
    start_date = triplet-triplet_delta

    frame1 = np.load(file_paths[start_date])
    #frame1 = ds[var].sel(time=start_date).values
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

    file_paths.pop(start_date, None)

    for date in file_paths:
        dates.append(date)
        print('flow calculation for date: ' + str(date))

        file = file_paths[date]
        frame2 = np.load(file)
        #frame2 = ds[var].sel(time=date).values

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

        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        path = '../data/processed/flow_frames/'
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = path + var + '_'+filename+'.npy'
        np.save(file_path, flow)
        file_paths_flow[date] = file_path
        prvs = next_frame
        shape = list(np.shape(frame2))
        shape.append(2)
        shape = tuple(shape)

    path = '../data/interim/dictionaries_optical_flow'
    if not os.path.exists(path):
        os.makedirs(path)
    file_dictionary = open(path+'/'+var+'.pkl', "wb")
    pickle.dump(file_paths_flow, file_dictionary)
