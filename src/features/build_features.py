#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:18:16 2019

@author: amirouyed
"""

import os
import pickle
import numpy as np


def builder(directory):
    """create qvdens from qv and airdens"""
    file_paths_airdens = pickle.load(
        open('../data/interim/dictionaries/AIRDENS.pkl', 'rb'))
    file_paths_qv = pickle.load(
        open('../data/interim/dictionaries/QV.pkl', 'rb'))
    file_paths_qvdens = {}
    for date in file_paths_airdens:
        frame1 = np.load(file_paths_airdens[date])
        frame2 = np.load(file_paths_qv[date])
        frame3 = np.multiply(frame1, frame2)
        file_path = str(directory+'/'+str(date)+".npy")
        np.save(file_path, frame3)
        file_paths_qvdens[date] = file_path
    dictionary_path = '../data/interim/dictionaries'
    file = open(dictionary_path+'/QVDENS.pkl', "wb")
    pickle.dump(file_paths_qvdens, file)
