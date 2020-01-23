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


def coarse_flow(flow,  pyr_scale, levels, iterations, poly_n, poly_sigma, prvs, next_frame, grid, coarse_grid):
    flowd = np.zeros(flow.shape)
    factor = grid/coarse_grid
    resized_prvs = cv2.resize(prvs, None, fx=factor, fy=factor)
    resized_next = cv2.resize(next_frame, None, fx=factor, fy=factor)
    flowx = flowd[:, :, 0]
    flowy = flowd[:, :, 1]
    resized_flowx = cv2.resize(flowx, None, fx=factor, fy=factor)
    resized_flowy = cv2.resize(flowy, None, fx=factor, fy=factor)
    shape = (resized_flowx.shape[0], resized_flowx.shape[1], 2)
    print('Flow coarsened from shape: ' +
          str(flow.shape) + ' to shape: ' + str(shape))
    resized_flow = np.zeros(shape)
    resized_flow[:, :, 0] = resized_flowx
    resized_flow[:, :, 1] = resized_flowy
    winsizes_small = [200, 100, 50, 25, 12]
    factor = 1/factor

    resized_prvs, resized_flow = multiscale_farneback(
        resized_flow,  resized_prvs, resized_next, pyr_scale, levels, winsizes_small, iterations, poly_n, poly_sigma)
    flowd[:, :, 0] = cv2.resize(
        resized_flow[:, :, 0], None, fx=factor, fy=factor)
    flowd[:, :, 1] = cv2.resize(
        resized_flow[:, :, 1], None, fx=factor, fy=factor)
    prvs = warp_flow(prvs, flowd)
    flow = flow+flowd
    return prvs, flow


def multiscale_farneback(flow,  prvs, next_frame, pyr_scale, levels, winsizes, iterations, poly_n, poly_sigma):
    for winsize in winsizes:
        flowd0 = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow = flow + flowd0
        prvs = warp_flow(prvs, flowd0)
    return prvs, flow


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    flow = flow.astype(np.float32)
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC)
    return res


def dof_averager(flow_x0, flow_y0, shape):
    leftovers = [0]*2
    frame_shape = np.shape(flow_x0)
    flowx = np.zeros(flow_x0.shape)
    flowy = np.zeros(flow_y0.shape)
    leftovers[0] = frame_shape[0] % shape[0]
    leftovers[1] = frame_shape[1] % shape[1]

    if(leftovers[0] != 0 and leftovers[1] != 0):
        flow_x0_view = flow_x0[:-leftovers[0], :-leftovers[1]]
        flow_y0_view = flow_y0[:-leftovers[0], :-leftovers[1]]

        flow_viewx = flowx[:-leftovers[0], :-leftovers[1]]
        flow_viewy = flowy[:-leftovers[0], :-leftovers[1]]
    elif (leftovers[0] == 0 and leftovers[1] != 0):
        flow_x0_view = flow_x0[:, :-leftovers[1]]
        flow_y0_view = flow_y0[:, :-leftovers[1]]

        flow_viewx = flowx[:, :-leftovers[1]]
        flow_viewy = flowy[:, :-leftovers[1]]
    elif (leftovers[0] != 0 and leftovers[1] == 0):
        flow_x0_view = flow_x0[:-leftovers[0], :]
        flow_y0_view = flow_y0[:-leftovers[0], :]
        flow_viewx = flowx[:-leftovers[0], :]
        flow_viewy = flowy[:-leftovers[0], :]
    else:
        flow_x0_view = flow_x0
        flow_y0_view = flow_y0
        flow_viewx = flowx
        flow_viewy = flowy
    patches_x = util.view_as_blocks(flow_x0_view, shape)
    patches_y = util.view_as_blocks(flow_y0_view, shape)
    shape = list(np.shape(flow_x0))
    shape.append(2)
    shape = tuple(shape)
    flow = np.zeros(shape)
    rows = patches_x.shape[0]
    cols = patches_x.shape[1]
    mean_strides_x = np.zeros((rows, cols))
    mean_strides_y = np.zeros((rows, cols))
    for x in range(0, rows):
        for y in range(0, cols):
            mean_strides_x[x, y] = np.mean(patches_x[x, y, ...])
            mean_strides_y[x, y] = np.mean(patches_y[x, y, ...])

    shape_inter = (flowx.shape[1], flowx.shape[0])
    flowx = cv2.resize(mean_strides_x, shape_inter, cv2.INTER_CUBIC)
    flowy = cv2.resize(mean_strides_y, shape_inter, cv2.INTER_CUBIC)
    flow[..., 0] = flowx
    flow[..., 1] = flowy

    return flow


def smoother(flowx, flowy):
    sizex = flowx.shape[1]
    sizey = flowx.shape[0]
    shape = (sizex, sizey)
    small_shape = (int(sizex/2), int(sizey/2))
    shapef = list(flowx.shape)
    shapef.append(2)
    flow = np.zeros(shapef)

    flowx = cv2.resize(flowx, small_shape, cv2.INTER_CUBIC)
    flowy = cv2.resize(flowy, small_shape, cv2.INTER_CUBIC)
    flowx = cv2.resize(flowx, shape,  cv2.INTER_CUBIC)
    flowy = cv2.resize(flowy, shape, cv2.INTER_CUBIC)
    flow[..., 0] = flowx
    flow[..., 1] = flowy

    return flow


def drop_nan(frame):
    row_mean = np.nanmean(frame, axis=1)
    inds = np.where(np.isnan(frame))
    frame[inds] = np.take(row_mean, inds[0])
    return frame


def optical_flow(start_date, var, pyr_scale, levels, iterations, poly_n, poly_sigma, sub_pixel, target_box_x, target_box_y, average_lon, tvl1, do_cross_correlation, farneback, stride_n, dof_average_x, dof_average_y, cc_average_x, cc_average_y, winsizes, grid, Lambda, coarse_grid,  **kwargs):
    """Implements cross correlation algorithm for calculating AMVs."""
    file_paths = pickle.load(
        open('../data/interim/dictionaries/vars/'+var+'.pkl', 'rb'))
    frame1 = np.load(file_paths[start_date])
    frame1 = drop_nan(frame1)
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    prvs = frame1
    file_paths_flow = {}
    file_paths.pop(start_date, None)
    for date in file_paths:
        print('flow calculation for date: ' + str(date))
        file = file_paths[date]
        frame2 = np.load(file)
        frame2 = drop_nan(frame2)

        frame2 = cv2.normalize(src=frame2, dst=None,
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_8UC1)
        next_frame = frame2
        shape = list(np.shape(frame2))
        shape.append(2)
        shape = tuple(shape)
        flow = np.zeros(shape)

        if tvl1:
            print('Initializing TV-L1 algorithm...')
            if coarse_grid > grid:
                prvs,  flow = coarse_flow(
                    flow,  pyr_scale, levels, iterations, poly_n, poly_sigma, prvs, next_frame, grid, coarse_grid)
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            optical_flow.setLambda(Lambda)
            flow = flow+optical_flow.calc(prvs, next_frame, None)
        if farneback:
            print('Initializing Farnebacks algorithm...')
            if coarse_grid > grid:
                prvs, flow = coarse_flow(
                    flow,  pyr_scale, levels, iterations, poly_n, poly_sigma, prvs, next_frame, grid, coarse_grid)

            prvs, flow = multiscale_farneback(
                flow,  prvs, next_frame, pyr_scale, levels, winsizes, iterations, poly_n, poly_sigma)

        if do_cross_correlation:
            print("Initializing cross correlation...")
            target_boxes = zip(target_box_x, target_box_x)
            for box in target_boxes:
                flowd0 = cc.amv_calculator(prvs, next_frame, box,
                                           sub_pixel, average_lon, int(stride_n))
                flow = flow + flowd0
                prvs = warp_flow(prvs, flowd0)

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
