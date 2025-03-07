

import cv2
from skimage import util
import numpy as np
import pickle
from viz import dataframe_calculators as dfc
import os


def vorticity_correction(start_date, var, pyr_scale, levels, iterations, poly_n, poly_sigma, sub_pixel, target_box_x, target_box_y, average_lon, tvl1, do_cross_correlation, farneback, stride_n, dof_average_x, dof_average_y, cc_average_x, cc_average_y, winsizes, grid, Lambda, coarse_grid, pyramid_factor, dt, file_paths, file_paths_flow, dates, **kwargs):

    ######
    frame1 = np.load(file_paths_flow[dates[0]])
    frame1 = drop_nan(frame1)
    vel = frame1
    print('calculating vorticity loop')
    frame1 = dfc.initial_vorticity(frame1, grid, 1/dt)
    frame1 = cv2.normalize(src=frame1, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

####
    frame1_qv = np.load(file_paths[dates[0]])
    frame1_qv = drop_nan(frame1_qv)
    frame1_qv = cv2.normalize(src=frame1_qv, dst=None,
                              alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    prvs = frame1
    prvs_qv = frame1_qv
    sdate = dates[0]
    dates.pop(0)
    for date in dates:
        print('vorticity flow calculation for date: ' + str(date))
        file = file_paths_flow[date]
        frame2 = np.load(file)
        frame2 = drop_nan(frame2)
        vel_next = frame2
        frame2 = dfc.initial_vorticity(frame2, grid, 1/dt)

        frame2 = cv2.normalize(src=frame2, dst=None, alpha=0,
                               beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        ####
        frame2_qv = np.load(file_paths[date])
        frame2_qv = drop_nan(frame2_qv)
        frame2_qv = cv2.normalize(src=frame2_qv, dst=None,
                                  alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        next_frame = frame2
        next_frame_qv = frame2_qv
        shape = list(np.shape(frame2))
        shape.append(2)
        shape = tuple(shape)

        #flow = vel

        print('Initializing Farnebacks algorithm...')

        optical_flow = cv2.optflow.createOptFlow_DeepFlow()
        flowd = optical_flow.calc(prvs, next_frame, None)
        flow = flowd
        prvs_qv = warp_flow(prvs_qv, flow)
        flowd = optical_flow.calc(prvs_qv, next_frame_qv, None)
        flow = flow+flowd

        file = file_paths[date]
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        file_path = '../data/processed/flow_frames/'+var+'_'+filename+'.npy'
        np.save(file_path, flow)
        file_paths_flow[date] = file_path
        prvs = next_frame
        prvs_qv = next_frame_qv
        vel = vel_next
    path = '../data/interim/dictionaries_optical_flow'

    if not os.path.exists(path):
        os.makedirs(path)

    file_dictionary = open(path+'/'+var+'.pkl', "wb")
    pickle.dump(file_paths_flow, file_dictionary)


def winsizes_creator(winsizes, pyramid_factor):
    while winsizes[-1] >= 10:
        winsizes.append(int(round(winsizes[-1]/pyramid_factor)))
    return winsizes


def tlv1_flow(flow,  prvs, next_frame,  Lambda):
    print('Initializing TV-L1 algorithm...')
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    optical_flow.setLambda(Lambda)
    flowd = optical_flow.calc(prvs, next_frame, None)
    flow = flow+flowd
    prvs = warp_flow(prvs, flowd)
    return prvs, flow


def coarse_flow_deep(flow,  prvs, next_frame, grid, coarse_grid):
    flowd = np.zeros(flow.shape)
    factor = grid/coarse_grid
    factor2 = coarse_grid/0.25
    resized_prvs = cv2.resize(prvs, None, fx=factor,
                              fy=factor, interpolation=cv2.INTER_CUBIC)
    resized_next = cv2.resize(
        next_frame, None, fx=factor, fy=factor,  interpolation=cv2.INTER_CUBIC)
    flowx = flowd[:, :, 0]
    flowy = flowd[:, :, 1]
    resized_flowx = cv2.resize(
        flowx, None, fx=factor, fy=factor,  interpolation=cv2.INTER_CUBIC)
    resized_flowy = cv2.resize(
        flowy, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    shape = (resized_flowx.shape[0], resized_flowx.shape[1], 2)
    print('Flow coarsened from shape: ' +
          str(flow.shape) + ' to shape: ' + str(shape))
    resized_flow = np.zeros(shape)
    resized_flow[:, :, 0] = resized_flowx
    resized_flow[:, :, 1] = resized_flowy

    factor = 1/factor
    optical_flow = cv2.optflow.createOptFlow_DeepFlow()

    resized_flow = optical_flow.calc(resized_prvs, resized_next, None)
    shape0 = (flow.shape[1], flow.shape[0])
    flowd[:, :, 0] = cv2.resize(
        resized_flow[:, :, 0], shape0)
    flowd[:, :, 1] = cv2.resize(
        resized_flow[:, :, 1], shape0)
    prvs = warp_flow(prvs, flowd)
    flow = flow+flowd
    print('frame succesfully warped with coarsened flow.')
    print('prvs mean: ' + str(np.mean(prvs)))
    return prvs, flow


def coarse_flow(flow,  pyr_scale, levels, iterations, poly_n, poly_sigma, prvs, next_frame, grid, coarse_grid, flag, winsizes_small):
    flowd = np.zeros(flow.shape)
    factor = grid/coarse_grid
    factor2 = coarse_grid/0.25
    resized_prvs = cv2.resize(prvs, None, fx=factor,
                              fy=factor, interpolation=cv2.INTER_CUBIC)
    resized_next = cv2.resize(
        next_frame, None, fx=factor, fy=factor,  interpolation=cv2.INTER_CUBIC)
    flowx = flowd[:, :, 0]
    flowy = flowd[:, :, 1]
    resized_flowx = cv2.resize(
        flowx, None, fx=factor, fy=factor,  interpolation=cv2.INTER_CUBIC)
    resized_flowy = cv2.resize(
        flowy, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    shape = (resized_flowx.shape[0], resized_flowx.shape[1], 2)
    print('Flow coarsened from shape: ' +
          str(flow.shape) + ' to shape: ' + str(shape))
    resized_flow = np.zeros(shape)
    resized_flow[:, :, 0] = resized_flowx
    resized_flow[:, :, 1] = resized_flowy

    factor = 1/factor
    resized_prvs, resized_flow = multiscale_farneback(
        resized_flow,  resized_prvs, resized_next, pyr_scale, levels, winsizes_small, iterations, poly_n, poly_sigma, flag)
    shape0 = (flow.shape[1], flow.shape[0])
    flowd[:, :, 0] = cv2.resize(
        resized_flow[:, :, 0], shape0)
    flowd[:, :, 1] = cv2.resize(
        resized_flow[:, :, 1], shape0)
    prvs = warp_flow(prvs, flowd)
    flow = flow+flowd
    print('frame succesfully warped with coarsened flow.')
    print('prvs mean: ' + str(np.mean(prvs)))
    return prvs, flow


def pyramid(flow, grid, coarse_grid, prvs, next_frame,  pyr_scale, levels, winsizes, iterations, poly_n, poly_sigma, pyramid_factor, Lambda):

    while grid < coarse_grid:
        prvs, flow = coarse_flow(flow,  pyr_scale, levels, iterations, poly_n, poly_sigma,
                                 prvs, next_frame, grid, coarse_grid, 0, winsizes)
       # if coarse_grid > 0.124 and coarse_grid <0.126:
        #   winsizes=[88,68,52,40,20,10]
        winsizes.insert(0, int(round(pyramid_factor*winsizes[0])))
        coarse_grid = coarse_grid/pyramid_factor

# test pyramid
    print('multiscale flow processing ...')

    prvs, flow = multiscale_farneback(
        flow,  prvs, next_frame, pyr_scale, levels, winsizes, iterations, poly_n, poly_sigma, 0)

    return prvs, flow, winsizes


def multiscale_farneback(flow,  prvs, next_frame, pyr_scale, levels, winsizes, iterations, poly_n, poly_sigma, flag):
    print('running pyramid with different filters for flag: ' + str(flag))
    for winsize in winsizes:
        print('running optical flow algorithm with fliter window: ' + str(winsize))
        flowd0 = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flag)
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
    #row_mean = np.nanmean(frame, axis=1)
    #inds = np.where(np.isnan(frame))
    #frame[inds] = np.take(row_mean, inds[0])
    mask = np.ma.masked_invalid(frame)
    mask = np.uint8(mask.mask)
    frame = np.nan_to_num(frame)

    frame = cv2.inpaint(frame, mask, inpaintRadius=10, flags=cv2.INPAINT_NS)
    print('inpainted')
    return frame
