#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:18:16 2019

@author: amirouyed
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def builder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_paths_AIRDENS = pickle.load(open('dictionaries/AIRDENS.pkl', 'rb'))
    file_paths_QV = pickle.load(open('dictionaries/QV.pkl', 'rb'))
    file_paths_QVDENS={}
    for date in file_paths_AIRDENS: 
        frame1 = np.load(file_paths_AIRDENS[date])
        frame2 = np.load(file_paths_QV[date])
        frame3=np.multiply(frame1,frame2)
        file_path=str(directory+'/'+str(date)+".npy")
        np.save(file_path,frame3)
        file_paths_QVDENS[date]=file_path
    dictionary_path='dictionaries'
    f = open(dictionary_path+'/QVDENS.pkl',"wb")
    pickle.dump(file_paths_QVDENS,f)
        