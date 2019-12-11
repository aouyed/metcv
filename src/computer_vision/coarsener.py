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
import glob

def coarsener(grid,):
    """Implements Farneback's optical flow algorithm."""
    
    dictionary_list=glob.glob('../data/interim/dictionaries/vars/*') 
    file_paths_list=[]
    for dict_path in dictionary_list:
          file_path_unit = pickle.load(open(dict_path 'rb'))
          file_paths_list.append(file_path_unit)        
    file_paths_coarse={}
    for file_path in file_paths
        for date in file_paths:
            file = file_paths[date]
            frame= np.load(file)
            factor=0.0625/grid
            resized_frame = cv2.resize(frame,None,fx=factor,fy=factor)
            filename = os.path.basename(file)
            filename = os.path.splitext(filename)[0]
            coarse_path='/data/processed/'+ str(grid)+'/'
               if not os.path.exists(path):
            os.makedirs(path)
            file_path = coarse_path+filename+'.npy'
            np.save(file_path, frame)
            file_paths_coarse[date] = file_path
        path = '../data/interim/dictionaries/coarsened_files'
        if not os.path.exists(path):
            os.makedirs(path)
        file_dictionary = open(path+'/'+var+'.pkl', "wb")
        pickle.dump(file_paths_flow, file_dictionary)
