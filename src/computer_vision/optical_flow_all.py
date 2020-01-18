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


def dof_averager(flow_x0,flow_y0, shape):
    leftovers=[0]*2
    frame_shape=np.shape(flow_x0)
    flowx=np.zeros(flow_x0.shape)
    flowy=np.zeros(flow_y0.shape)
    leftovers[0]=frame_shape[0]%shape[0]
    leftovers[1]=frame_shape[1]%shape[1]
       
    if(leftovers[0]!=0 and leftovers[1]!=0):
        flow_x0_view=flow_x0[:-leftovers[0],:-leftovers[1]]
        flow_y0_view=flow_y0[:-leftovers[0],:-leftovers[1]]

        flow_viewx=flowx[:-leftovers[0],:-leftovers[1]]
        flow_viewy=flowy[:-leftovers[0],:-leftovers[1]]
    elif (leftovers[0]==0 and leftovers[1]!=0):
        flow_x0_view=flow_x0[:,:-leftovers[1]]
        flow_y0_view=flow_y0[:,:-leftovers[1]]

        flow_viewx=flowx[:,:-leftovers[1]]
        flow_viewy=flowy[:,:-leftovers[1]]
    elif (leftovers[0]!=0 and leftovers[1]==0):
        flow_x0_view=flow_x0[:-leftovers[0],:]
        flow_y0_view=flow_y0[:-leftovers[0],:]
        flow_viewx=flowx[:-leftovers[0],:]
        flow_viewy=flowy[:-leftovers[0],:]
    else:
        flow_x0_view=flow_x0
        flow_y0_view=flow_y0
        flow_viewx=flowx
        flow_viewy=flowy
    patches_x=util.view_as_blocks(flow_x0_view, shape)
    patches_y=util.view_as_blocks(flow_y0_view, shape)
    shape=list(np.shape(flow_x0))
    shape.append(2)
    shape=tuple(shape)
    flow=np.zeros(shape)
    rows = patches_x.shape[0]
    cols = patches_x.shape[1]
    mean_strides_x=np.zeros((rows,cols))
    mean_strides_y=np.zeros((rows,cols))
    for x in range(0, rows):
        for y in range(0, cols):
            mean_strides_x[x,y]=np.mean(patches_x[x,y,...])
            mean_strides_y[x,y]=np.mean(patches_y[x,y,...])

    shape_inter=(flowx.shape[1],flowx.shape[0])
    flowx=cv2.resize(mean_strides_x,shape_inter, cv2.INTER_CUBIC)
    flowy=cv2.resize(mean_strides_y,shape_inter, cv2.INTER_CUBIC)
    flow[...,0]= flowx
    flow[...,1]= flowy


    return flow

def smoother(flowx,flowy):
    sizex=flowx.shape[1]
    sizey=flowx.shape[0]
    shape=(sizex,sizey)
    small_shape=(int(sizex/2),int(sizey/2))
    shapef=list(flowx.shape)
    shapef.append(2)
    flow=np.zeros(shapef)
   
    flowx=cv2.resize(flowx,small_shape, cv2.INTER_CUBIC)
    flowy=cv2.resize(flowy,small_shape, cv2.INTER_CUBIC)
    flowx=cv2.resize(flowx,shape,  cv2.INTER_CUBIC)
    flowy=cv2.resize(flowy,shape, cv2.INTER_CUBIC)
    flow[...,0]= flowx
    flow[...,1]= flowy


    return flow



def drop_nan(frame):
    row_mean = np.nanmean(frame, axis=1)
    inds = np.where(np.isnan(frame))
    frame[inds] = np.take(row_mean, inds[0])   
    return frame 


def optical_flow(start_date, var, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, sub_pixel, target_box_x,target_box_y, average_lon, tvl1, do_cross_correlation, farneback, stride_n, dof_average_x, dof_average_y, cc_average_x, cc_average_y, **kwargs):
    """Implements cross correlation algorithm for calculating AMVs."""
    file_paths = pickle.load(
        open('../data/interim/dictionaries/vars/'+var+'.pkl', 'rb'))
    frame1 = np.load(file_paths[start_date])
    frame1=drop_nan(frame1)
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    prvs = frame1
    file_paths_flow = {}
    file_paths.pop(start_date, None)
    for date in file_paths:
        print('cross correlation calculation for date: ' + str(date))
        file = file_paths[date]
        frame2 = np.load(file)
        frame2=drop_nan(frame2)

        frame2 = cv2.normalize(src=frame2, dst=None,
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_8UC1)
        next_frame = frame2
        shape=list(np.shape(frame2))
        shape.append(2)
        shape=tuple(shape)
        flow=np.zeros(shape)
        
        
        if do_cross_correlation:
            target_boxes=zip(target_box_y,target_box_x)
            for box in target_boxes:
                #flow =flow+cc. amv_calculator(prvs, next_frame,(4*box[0],4*box[1]), sub_pixel, average_lon, int(stride_n/2))
                flow0 =flow+cc. amv_calculator(prvs, next_frame,box, sub_pixel, average_lon, int(stride_n))
                flow=dof_averager(flow0[...,0],flow0[...,1],(cc_average_y,cc_average_x)) + dof_averager(flow0[...,0],flow0[...,1],(int(cc_average_y/2),int(cc_average_x/2)))
                #flow=smoother(flow0[...,0],flow0[...,1])
                
        if tvl1:
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            optical_flow.setLambda(0.005)
            flow = flow+optical_flow.calc(prvs, next_frame, None)
        if farneback:
            flowd=cv2.calcOpticalFlowFarneback(prvs, next_frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            if dof_average_x> 1 or dof_average_y>1:            flowd0=dof_averager(flowd[...,0],flowd[...,1],(dof_average_y,dof_average_y))
            else:
                flowd0=np.zeros(flow.shape)
            #flowd1=dof_averager(flowd[...,0],flowd[...,1],(20,20))
            flow =flow+ flowd0
 
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

