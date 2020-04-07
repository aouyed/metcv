#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:56:34 2019

@author: aouyed
"""

import glob
import pickle
from computer_vision import optical_flow_calculators as ofc
import cv2
from datetime import timedelta
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import metpy.calc as mpcalc
from scipy.interpolate import bisplrep
from scipy.interpolate import bisplev
import math
from tqdm import tqdm
from numba import jit
import metpy
from metpy.units import units
import scipy.ndimage as ndimage


def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram,  modified from
    https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    """

    fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, normed=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
                data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    im = ax.scatter(x, y, c=z, **kwargs)
    fig.colorbar(im, ax=ax)
    return ax


def daterange(start_date, end_date):
    """creates a list of dates to be iterated"""
    date_list = []
    delta = timedelta(hours=1)
    while start_date < end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


def dataframe_quantum(file, date, dictionary_dict):
    """creates a datafrane for a particular date meant for being concatenated"""
    frame = np.load(file)

    df = pd.DataFrame(frame[:, :, 0]).stack().rename_axis(
        ['y', 'x']).reset_index(name='flow_u')
    df_1 = pd.DataFrame(frame[:, :, 1]).stack().rename_axis(
        ['y', 'x']).reset_index(name='flow_v')
    df['flow_v'] = df_1['flow_v']
    df['datetime'] = pd.Timestamp(date)
    for state_var in dictionary_dict:
        state_files = dictionary_dict[state_var]
        frame = np.load(state_files[date])
        df_1 = pd.DataFrame(frame).stack().rename_axis(
            ['y', 'x']).reset_index(name=state_var.lower())
        df = df.merge(df_1, how='left')
    return df


def dataframe_pivot(frame, var):
    df = pd.DataFrame(frame).stack().rename_axis(
        ['y', 'x']).reset_index(name=var.lower())
    return df


def scaling_df_approx(df, grid, dt_inv):
    """coordinate transforms vels from angle/pixel to metric, approximately"""

    df['u_scaled_approx'] = scaling_u(
        df['lon'], df['lat'], df['flow_u'], grid, dt_inv)
    df['v_scaled_approx'] = scaling_v(
        df['lon'], df['lat'], df['flow_v'], grid, dt_inv)

    return df


def scaling_u(df_lon, df_lat, df_flow_u, grid, dt_inv):
    dtheta = grid*df_flow_u
    drads = dtheta * math.pi / 180
    lat = df_lat*math.pi/90/2
    # dt_hr = 1
    # dt_s = 3600
    R = 6371000
    scaleConstant = dt_inv
    dx = R*abs(np.cos(lat))*drads
    scale = dx*scaleConstant
    return scale


def scaling_v(df_lon, df_lat, df_flow_v, grid, dt_inv):
    """coordinate transform for v from pixel/angular to metric, approximate"""
    dtheta = grid*df_flow_v
    drads = dtheta * math.pi / 180
    R = 6371000
    scaleConstant = dt_inv
    dx = R*drads
    scale = dx*scaleConstant
    return scale


def error_calculator(df):
    """calculates error between ground truth and opticla flow values"""
    df["error_u"] = df['u']-df['flow_u']
    df["error_v"] = df['v']-df['flow_v']
    df["error_u_norm"] = df["error_u"]/df['u']
    df["error_v_norm"] = df["error_v"]/df['v']

    return df


def latlon_converter(df, dtheta):
    """coordinate transform for pixel to angular"""

    df['lat'] = df['y']*dtheta - 90
    df['lon'] = df['x']*dtheta - 180
    return df
