#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:15:17 2019

@author: amirouyed
"""

import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from skimage.feature import register_translation

def amv_calculator(prvs_frame, next_frame):
    prvs_patches=image.extract_patches_2d(prvs_frame, (10, 10))
    next_patches = image.extract_patches_2d(next_frame, (10, 10))
    shape=list(np.shape(prvs_frame))
    shape.append(2)
    shape=tuple(shape)
    flow=np.zeros(shape)
    shift_patches_x=np.zeros(np.shape(prvs_patches))
    shift_patches_y=np.zeros(np.shape(prvs_patches))

    for i,next_patch in enumerate(next_patches):
        shift, error, diffphase = register_translation(prvs_patches[i], next_patch)
        shift_patches_x[i]=shift[1]
        shift_patches_y[i]=shift[0]

        
    flowx= image.reconstruct_from_patches_2d(shift_patches_x, np.shape(next_frame))
    flowy= image.reconstruct_from_patches_2d(shift_patches_y, np.shape(next_frame))
    flow[...,0]=flowx
    flow[...,1]=flowy

    return flow

def cross_correlation_amv(start_date, var, pyr_scale, levels, winsize, iterations,
                            poly_n, poly_sigma,**kwargs):
    """Implements cross correlation algorithm for calculating AMVs."""
    file_paths = pickle.load(
        open('../data/interim/dictionaries/vars/'+var+'.pkl', 'rb'))
    frame1 = np.load(file_paths[start_date])
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    prvs = frame1
    file_paths_flow = {}
    file_paths.pop(start_date, None)
    for date in file_paths:
        file = file_paths[date]
        frame2 = np.load(file)
        frame2 = cv2.normalize(src=frame2, dst=None,
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_8UC1)
        next_frame = frame2
        flow = amv_calculator(prvs, next_frame)
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        file_path = '../data/processed/flow_frames/'+var+'_'+filename+'.npy'
        np.save(file_path, flow)
        file_paths_flow[date] = file_path
        prvs = next_frame
    cv2.destroyAllWindows()
    path = '../data/interim/dictionaries_optical_flow'
    if not os.path.exists(path):
        os.makedirs(path)
    file_dictionary = open(path+'/'+var+'.pkl', "wb")
    pickle.dump(file_paths_flow, file_dictionary)