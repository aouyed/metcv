#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:18:16 2019

@author: amirouyed
"""

import os
import pickle
import numpy as np


def builder(var):
    """create qvdens from qv and airdens"""
    file_paths_airdens = pickle.load(
        open('../data/interim/dictionaries/vars/AIRDENS.pkl', 'rb'))
    file_paths_qv = pickle.load(
        open('../data/interim/dictionaries/vars/QV.pkl', 'rb'))
    file_paths_qvdens = {}
    directory_path = '../data/interim/'+var.lower()
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    for date in file_paths_airdens:
        frame1 = np.load(file_paths_airdens[date])
        frame2 = np.load(file_paths_qv[date])
        frame3 = np.multiply(frame1, frame2)
        file_path = str(directory_path+'/'+str(date)+".npy")
        np.save(file_path, frame3)
        file_paths_qvdens[date] = file_path
    dictionary_path = '../data/interim/dictionaries/vars'
    if not os.path.exists(dictionary_path):
        os.makedirs(dictionary_path)
    file = open(dictionary_path+'/QVDENS.pkl', "wb")
    pickle.dump(file_paths_qvdens, file)
