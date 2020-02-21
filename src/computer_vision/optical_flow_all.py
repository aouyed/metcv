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


def optical_flow(start_date, end_date, var, pyr_scale, levels, iterations, poly_n, poly_sigma, sub_pixel, target_box_x, target_box_y, average_lon, tvl1, do_cross_correlation, farneback, stride_n, dof_average_x, dof_average_y, cc_average_x, cc_average_y, winsizes, grid, Lambda, coarse_grid, pyramid_factor, dt, nudger, deep_flow, jpl_loader,   **kwargs):
    """Implements cross correlation algorithm for calculating AMVs."""

    file_paths = pickle.load(
        open('../data/interim/dictionaries/vars/'+var+'.pkl', 'rb'))
    file_paths_u = pickle.load(
        open('../data/interim/dictionaries/vars/u.pkl', 'rb'))
    file_paths_v = pickle.load(
        open('../data/interim/dictionaries/vars/v.pkl', 'rb'))
    prvs_date = end_date - timedelta(hours=1)
    factor_flowu = 1
    factor_flowv = 1
    if jpl_loader:
        prvs_date = start_date
    frame1 = np.load(file_paths[prvs_date])

    frame1 = ofc.drop_nan(frame1)
    frame1=np.nan_to_num(frame1)
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    ####

    # frame1 = cv2.equalizeHist(frame1)
    shape = list(np.shape(frame1))
    shape.append(2)
    shape = tuple(shape)
    flow = np.zeros(shape)
    if nudger:
        frame1_u = np.load(file_paths_u[start_date])
        frame1_v = np.load(file_paths_v[start_date])
        flow[:, :, 0] = frame1_u
        flow[:, :, 1] = frame1_v
        flow[:, :, 0] = 0.25
        flow[:, :, 1] = 0.25
        flow = dfc.initial_flow(flow, grid, 1/dt)
        flow = np.nan_to_num(flow)
        #flow[:, :, 0] = 0.5*flow[:, :, 0]
        #flow[:, :, 1] = 0.6*flow[:, :, 1]
        flow[:, :, 0] = factor_flowu*flow[:, :, 0]
        flow[:, :, 1] = factor_flowv*flow[:, :, 1]

    prvs = frame1
    prvs = np.nan_to_num(prvs)
    file_paths_flow = {}
    dates = []
    file_paths.pop(start_date, None)
    file_paths_e = {}
    file_paths_e[end_date] = file_paths[end_date]
    if not jpl_loader:
        file_paths = file_paths_e

    for date in file_paths:
        print('warping...')
        print('nudger value')
        print(nudger)
        if nudger:
            # helper variable to avoid random segfault
            prvsh = ofc.warp_flow(prvs, flow)
            prvs = prvsh
       # prvs = ofc.warp_flow(prvs, flow)

        dates.append(date)
        print('flow calculation for date: ' + str(date))
        file = file_paths[date]
        frame2 = np.load(file)
        frame2 = ofc.drop_nan(frame2)
        # frame2 = np.nan_to_num(frame2)
        frame2 = cv2.normalize(src=frame2, dst=None, alpha=0,
                               beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        next_frame = frame2

        print('Initializing Farnebacks algorithm...')

        print('segmentation fault?')

        if deep_flow:
            optical_flow = cv2.optflow.createOptFlow_DeepFlow()
            flowd = optical_flow.calc(prvs, next_frame, None)

            flow = flow+flowd
            if not nudger:
                prvs = ofc.warp_flow(prvs, flow)
                flowd = optical_flow.calc(prvs, next_frame, None)
                flow = flow+flowd
            #flow = np.zeros(shape)

       # optical_flow = cv2.optflow.createOptFlow_DeepFlow()
       # flowv=optical_flow.calc(prvs0, next_frame, None)

        print('done with deep flow')
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        file_path = '../data/processed/flow_frames/'+var+'_'+filename+'.npy'
        np.save(file_path, flow)
        file_paths_flow[date] = file_path
        prvs = next_frame
        frame1_u = np.load(file_paths_u[date])
        frame1_v = np.load(file_paths_v[date])
        # frame1 = cv2.equalizeHist(frame1)
        shape = list(np.shape(frame2))
        shape.append(2)
        shape = tuple(shape)
        flow = np.zeros(shape)
        if nudger:
           # flow[:, :, 0] = factor_flowu*frame1_u
            #flow[:, :, 1] = factor_flowv*frame1_v
            flow[:, :, 0] = 0.25
            flow[:, :, 1] = 0.25
            flow = dfc.initial_flow(flow, grid, 1/dt)
            flow = np.nan_to_num(flow)

    path = '../data/interim/dictionaries_optical_flow'
    if not os.path.exists(path):
        os.makedirs(path)
    file_dictionary = open(path+'/'+var+'.pkl', "wb")
    pickle.dump(file_paths_flow, file_dictionary)

#######

   # ofc.vorticity_correction(start_date, var, pyr_scale, levels, iterations, poly_n, poly_sigma, sub_pixel, target_box_x, target_box_y, average_lon, tvl1, do_cross_correlation, farneback,stride_n, dof_average_x, dof_average_y, cc_average_x, cc_average_y, winsizes, grid, Lambda, coarse_grid, pyramid_factor, dt, file_paths, file_paths_flow, dates, **kwargs)
