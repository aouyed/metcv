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


def optical_flow(start_date, var, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, sub_pixel, target_box, tvl1, do_cross_correlation, farneback, **kwargs):
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
        print('cross correlation calculation for date: ' + str(date))
        file = file_paths[date]
        frame2 = np.load(file)
        frame2 = cv2.normalize(src=frame2, dst=None,
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_8UC1)
        next_frame = frame2
        shape=list(np.shape(frame2))
        shape.append(2)
        shape=tuple(shape)
        flow=np.zeros(shape)
        
        if do_cross_correlation:
            for box in target_box:
                flow =flow+cc. amv_calculator(prvs, next_frame,(box,box), sub_pixel)
        if tvl1:
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            optical_flow.setLambda(0.005)
            flow = flow+optical_flow.calc(prvs, next_frame, None)
        if farneback:
            flow =flow+ cv2.calcOpticalFlowFarneback(prvs, next_frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            #flow[...,1]=cv2.fastNlMeansDenoising(np.uint8(flow[...,1]),None)
            #flow[...,0]=cv2.fastNlMeansDenoising(np.uint8(flow[...,0]),None)  
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
