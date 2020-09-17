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

R = 6371000


def daterange(start_date, end_date):
    """creates a list of dates to be iterated"""
    date_list = []
    delta = timedelta(hours=1)
    while start_date < end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


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
    lat = df_lat*math.pi/180
    dx = R*abs(np.cos(lat))*drads
    scale = dx*dt_inv
    return scale


def scaling_v(df_lon, df_lat, df_flow_v, grid, dt_inv):
    """coordinate transform for v from pixel/angular to metric, approximate"""
    dtheta = grid*df_flow_v
    drads = dtheta * math.pi / 180
    dx = R*drads
    scale = dx*dt_inv
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
