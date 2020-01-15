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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from skimage.feature import register_translation
from tqdm import trange

def amv_calculator(prvs_frame, next_frame,shape, sub_pixel):
    leftovers=[0]*2
    frame_shape=np.shape(prvs_frame)
    flowx=np.zeros(prvs_frame.shape)
    flowy=np.zeros(prvs_frame.shape)
    leftovers[0]=frame_shape[0]%shape[0]
    leftovers[1]=frame_shape[1]%shape[1]
       
    if(leftovers[0]!=0 and leftovers[1]!=0):
        prvs_frame_view=prvs_frame[:-leftovers[0],:-leftovers[1]]
        next_frame_view=next_frame[:-leftovers[0],:-leftovers[1]]
        flow_viewx=flowx[:-leftovers[0],:-leftovers[1]]
        flow_viewy=flowy[:-leftovers[0],:-leftovers[1]]
    elif (leftovers[0]==0 and leftovers[1]!=0):
        prvs_frame_view=prvs_frame[:,:-leftovers[1]]
        next_frame_view=next_frame[:,:-leftovers[1]]
        flow_viewx=flowx[:,:-leftovers[1]]
        flow_viewy=flowy[:,:-leftovers[1]]
    elif (leftovers[0]!=0 and leftovers[1]==0):
        prvs_frame_view=prvs_frame[:-leftovers[0],:]
        next_frame_view=next_frame[:-leftovers[0],:]
        flow_viewx=flowx[:-leftovers[0],:]
        flow_viewy=flowy[:-leftovers[0],:]
    else:
        prvs_frame_view=prvs_frame
        next_frame_view=next_frame
        flow_viewx=flowx
        flow_viewy=flowy
            
    print("shape of original frame: " + str(np.shape(prvs_frame)))

    print("shape of array view removing leftovers: " + str(np.shape(prvs_frame_view)))

    prvs_patches=util.view_as_blocks(prvs_frame_view, shape)
    next_patches = util.view_as_blocks(next_frame_view, shape)
    print("shape of array of blocks: " + str(np.shape(prvs_patches)))

    shift_patches_x=util.view_as_blocks(flow_viewx, shape)
    shift_patches_y=util.view_as_blocks(flow_viewy, shape)
 
    shape=list(np.shape(prvs_frame))
    shape.append(2)
    shape=tuple(shape)
    flow=np.zeros(shape)
    rows = prvs_patches.shape[0]
    cols = prvs_patches.shape[1]
    shift_inter_x=np.zeros((rows,cols))
    shift_inter_y=np.zeros((rows,cols))
    print('progress of cross correlation calculation:')
    for x in trange(0, rows):
        for y in range(0, cols):
            if not sub_pixel:
                shift, error, diffphase = register_translation(prvs_patches[x,y,...], next_patches[x,y,...])
            else:
                shift, error, diffphase = register_translation(prvs_patches[x,y,...], next_patches[x,y,...],100)

            shift_patches_x[x,y,...]=shift[1]
            shift_patches_y[x,y,...]=shift[0]
            shift_inter_x[x,y]=shift[1]
            shift_inter_y[x,y]=shift[0]
    print('mean pixel offset in x direction: ' + str(np.mean(flowx)))
    print('mean pixel offset in y direction: ' + str(np.mean(flowy)))
    shape_inter=(flowx.shape[1],flowx.shape[0])
    flowx=cv2.resize(shift_inter_x,shape_inter)
    flowy=cv2.resize(shift_inter_y,shape_inter)
    flow[...,0]= -flowx
    flow[...,1]= -flowy


    return flow

def cross_correlation_amv(start_date, var, pyr_scale, levels, winsize, iterations,
                            poly_n, poly_sigma, sub_pixel, target_box,**kwargs):
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
        flow = amv_calculator(prvs, next_frame,(target_box,target_box), sub_pixel)
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
