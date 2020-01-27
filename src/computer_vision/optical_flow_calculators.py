

import cv2
from skimage import util
import numpy as np


def winsizes_creator(winsizes, pyramid_factor):
    while winsizes[-1] >= 10:
        winsizes.append(int(round(winsizes[-1]/pyramid_factor)))
    return winsizes


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


def pyramid(flow, grid, coarse_grid, prvs, next_frame,  pyr_scale, levels, winsizes, iterations, poly_n, poly_sigma, pyramid_factor):

    while grid < coarse_grid:
        prvs, flow = coarse_flow(flow,  pyr_scale, levels, iterations, poly_n, poly_sigma,
                                 prvs, next_frame, grid, coarse_grid, 0, winsizes)
        winsizes.insert(0, int(round(pyramid_factor*winsizes[0])))
        coarse_grid = coarse_grid/pyramid_factor

    prvs, flow = multiscale_farneback(
        flow,  prvs, next_frame, pyr_scale, levels, winsizes, iterations, poly_n, poly_sigma, 0)

    flow[:, :, 0] = cv2.blur(flow[:, :, 0], (3, 3))
    flow[:, :, 1] = cv2.blur(flow[:, :, 1], (3, 3))

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
    row_mean = np.nanmean(frame, axis=1)
    inds = np.where(np.isnan(frame))
    frame[inds] = np.take(row_mean, inds[0])
    return frame
