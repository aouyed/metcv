#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:56:52 2019

@author: amirouyed
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import cv2

def frame_maker(var, path):
    
    directory='../data/processed/frames_'+ path
    file_paths = pickle.load(open('../data/interim/dictionaries/'+var+'.pkl', 'rb'))
    if not os.path.exists(directory):
        os.makedirs(directory)
    for date in file_paths:
        frame1 = np.load(file_paths[date])
        frame1 = cv2.normalize(src=frame1, dst=None,
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        plt.imsave(directory+'/'+str(date)+'.png',frame1)
