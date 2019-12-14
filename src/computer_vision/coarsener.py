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

def coarsener(grid,**kwargs):
    """Implements opencv resizing."""
    
    dictionary_list=glob.glob('../data/interim/dictionaries/vars/*') 
    for dict_path in dictionary_list:
        file_paths= pickle.load(open(dict_path, 'rb'))
        file_paths_coarse={}
        for date in file_paths:  
            filename=os.path.basename(dict_path)
            var=os.path.splitext(filename)[0]
            file = file_paths[date]
            frame= np.load(file)
            factor=0.0625/grid
            resized_frame = cv2.resize(frame,None,fx=factor,fy=factor)
            filename = os.path.basename(file)
            filename = os.path.splitext(filename)[0]
            coarse_path='../data/interim/'+ str(grid)+'/'
            if not os.path.exists(coarse_path):
                os.makedirs(coarse_path)
            file_path = coarse_path+filename+'.npy'
            np.save(file_path, resized_frame)
            file_paths_coarse[date] = file_path
        path = '../data/interim/dictionaries/vars'
        if not os.path.exists(path):
            os.makedirs(path)
        file_dictionary = open(path+'/'+var+'.pkl', "wb")
        pickle.dump(file_paths_coarse, file_dictionary)
        print('frame shape for var '+ str(var) +' at 0.0625 deg is: ' + str(frame.shape))
        print('frame shape for var '+ str(var) +' at ' + str(grid)+  'deg is: ' + str(resized_frame.shape))
