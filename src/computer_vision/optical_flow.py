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

def optical_flow_calculator(start_date, var,pyr_scale, levels, winsize, iterations, 
                            poly_n, poly_sigma):
    """Implements Farneback's optical flow algorithm."""
    file_paths = pickle.load(open('../../data/interim/dictionaries/'+var+'.pkl', 'rb'))
    frame1 = np.load(file_paths[start_date])
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    #plt.imsave('grey.png',frame1)
    prvs = frame1
    file_paths_flow = {}
    file_paths.pop(start_date, None)
    for date in file_paths:
        file = file_paths[date]
        frame2 = np.load(file)
        frame2 = cv2.normalize(src=frame2, dst=None,
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                               dtype=cv2.CV_8UC1)
        try:
            next = frame2
        except:
            print('no more frames')
            break
        #flow = cv2.calcOpticalFlowFarneback(prvs, next,
        #                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, 
                                            levels, winsize, iterations, 
                                            poly_n, poly_sigma, 0)
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        file_path = 'flow_frames/'+var+'_'+filename+'.npy'
        np.save(file_path, flow)
        file_paths_flow[date] = file_path
        prvs = next
    cv2.destroyAllWindows()
    path='../../data/interim/dictionaries_optical_flow'
    if not os.path.exists(path):
        os.makedirs(path)
    file_dictionary = open(path+'/'+var+'.pkl', "wb")
    pickle.dump(file_paths_flow, file_dictionary)
    