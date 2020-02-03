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


def optical_flow(start_date, var, pyr_scale, levels, iterations, poly_n, poly_sigma, sub_pixel, target_box_x, target_box_y, average_lon, tvl1, do_cross_correlation, farneback, stride_n, dof_average_x, dof_average_y, cc_average_x, cc_average_y, winsizes, grid, Lambda, coarse_grid, pyramid_factor, dt, **kwargs):
    """Implements cross correlation algorithm for calculating AMVs."""

    file_paths = pickle.load(
        open('../data/interim/dictionaries/vars/'+var+'.pkl', 'rb'))
    frame1 = np.load(file_paths[start_date])
    frame1 = ofc.drop_nan(frame1)
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # frame1 = cv2.equalizeHist(frame1)

    prvs = frame1
    file_paths_flow = {}
    dates = []
    file_paths.pop(start_date, None)
    for date in file_paths:
        dates.append(date)
        print('flow calculation for date: ' + str(date))
        file = file_paths[date]
        frame2 = np.load(file)
        frame2 = ofc.drop_nan(frame2)

        frame2 = cv2.normalize(src=frame2, dst=None, alpha=0,
                               beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # frame2 = cv2.equalizeHist(frame2)
        next_frame = frame2
        shape = list(np.shape(frame2))
        shape.append(2)
        shape = tuple(shape)
        flow = np.zeros(shape)

        print('Initializing Farnebacks algorithm...')

        prvs, flow = ofc.coarse_flow_deep(
            flow,  prvs, next_frame, grid, grid)

        optical_flow = cv2.optflow.createOptFlow_DeepFlow()

        flow = flow+optical_flow.calc(prvs, next_frame, None)
        print('done with deep flow')
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        file_path = '../data/processed/flow_frames/'+var+'_'+filename+'.npy'
        np.save(file_path, flow)
        file_paths_flow[date] = file_path
        prvs = next_frame

    path = '../data/interim/dictionaries_optical_flow'
    if not os.path.exists(path):
        os.makedirs(path)
    file_dictionary = open(path+'/'+var+'.pkl', "wb")
    pickle.dump(file_paths_flow, file_dictionary)

#######

    frame1 = np.load(file_paths_flow[dates[0]])
    frame1 = ofc.drop_nan(frame1)
    vel = frame1
    print('calculating vorticity loop')
    frame1 = dfc.initial_vorticity(frame1, grid, 1/dt)
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
   # frame1 = cv2.equalizeHist(frame1)

    prvs = frame1
    sdate = dates[0]
    dates.pop(0)
    for date in dates:
        print('vorticity flow calculation for date: ' + str(date))
        file = file_paths_flow[date]
        frame2 = np.load(file)
        frame2 = ofc.drop_nan(frame2)
        vel_next = frame2
        frame2 = dfc.initial_vorticity(frame2, grid, 1/dt)

        frame2 = cv2.normalize(src=frame2, dst=None, alpha=0,
                               beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # frame2 = cv2.equalizeHist(frame2)
        next_frame = frame2
        shape = list(np.shape(frame2))
        shape.append(2)
        shape = tuple(shape)
        flow = vel
        prvs = ofc.warp_flow(prvs, flow)

        print('Initializing Farnebacks algorithm...')

        optical_flow = cv2.optflow.createOptFlow_DeepFlow()

        flow = flow+optical_flow.calc(prvs, next_frame, None)
        file = file_paths[date]
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        file_path = '../data/processed/flow_frames/'+var+'_'+filename+'.npy'
        np.save(file_path, flow)
        file_paths_flow[date] = file_path
        prvs = next_frame
        vel = vel_next
    path = '../data/interim/dictionaries_optical_flow'

    if not os.path.exists(path):
        os.makedirs(path)

    file_dictionary = open(path+'/'+var+'.pkl', "wb")
    pickle.dump(file_paths_flow, file_dictionary)
