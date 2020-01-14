#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:54:27 2019

@author: aouyed
"""
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


def optical_flow_calculator(start_date, var, pyr_scale, levels, winsize, iterations,
                            poly_n, poly_sigma, tvl1,  **kwargs):
    """Implements Farneback's optical flow algorithm."""
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

        if tvl1:
           # optical_flow = cv2.DualTVL1OpticalFlow_create()
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            optical_flow.setLambda(0.005)
            flow = optical_flow.calc(prvs, next_frame, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, pyr_scale,
                                                levels, winsize, iterations,
                                                poly_n, poly_sigma, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        path = '../data/processed/flow_frames'
        file_path = '../data/processed/flow_frames/'+var+'_'+filename+'.npy'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(file_path, flow)
        file_paths_flow[date] = file_path
        prvs = next_frame
    cv2.destroyAllWindows()
    path = '../data/interim/dictionaries_optical_flow'
    if not os.path.exists(path):
        os.makedirs(path)
    file_dictionary = open(path+'/'+var+'.pkl', "wb")
    pickle.dump(file_paths_flow, file_dictionary)
