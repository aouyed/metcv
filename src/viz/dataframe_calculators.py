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


def heatmap_plotter(df, date, directory):
    """createp heatmaps through all dataframes columns"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    df = df.loc[[date]]
    exclude = ('x', 'y', 'lat', 'lon', 'datetime')
    for column in df:
        if not column in exclude:
            heatmapper(df, column, directory)


def heatmapper(df, values, directory):
    """creates a heatmap from a column"""

    piv = pd.pivot_table(df, values=values,
                         index=["lat"], columns=["lon"], fill_value=0)
    fig, ax = plt.subplots()
    im = ax.imshow(piv, cmap=sns.cm.rocket,
                   extent=[-180, 180, -90, 90], origin='lower')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('m/s')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(values)
    plt.tight_layout()
    plt.savefig(directory + '/'+values+'.png', bbox_inches='tight', dpi=1000)
    plt.close()


def plotter(df, directory, date):
    """creates scatter plots through various column combinations"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    df = df.loc[[date]]
    for column_a in df:
        for column_b in df:
            if column_a != column_b:
                # ax = df.plot(kind="scatter", x=column_a, y=column_b)
                # plt.hist2d(df[column_a], df[column_b], (500, 500), cmap=plt.cm.jet)
                # plt.colorbar()
                ax = density_scatter(df[column_a], df[column_b], bins=[
                                     30, 30], s=1, cmap=sns.cm.rocket)
                plt.savefig(directory + '/'+column_a+'_' +
                            column_b+'.png', bbox_inches='tight')
                plt.close()


def plot_average(deltax, df, xlist, varx, vary):
    df_mean = pd.DataFrame()
    df_unit = pd.DataFrame(data=[0], columns=[varx])
    print("calculating averages ...")
    for x in tqdm(xlist):
        df_a = df[df[varx] >= x]
        df_a = df_a[df_a[varx] <= x+deltax]
        df_unit[varx] = x
        df_unit[vary+'_count'] = df_a[vary].shape[0]
        df_unit[vary + '_std'] = df_a[vary].std()
        df_unit[vary] = df_a[vary].mean()
        if df_mean.empty:
            df_mean = df_unit
        else:
            df_mean = pd.concat([df_mean, df_unit])

    # df_mean.to_pickle(directory+'/df_mean_'+varx+'_'+vary+'.pkl')
    return df_mean


def line_plotter(df, directory):
    """creates scatter plots through various column combinations"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    for column_a in df:
        for column_b in df:
            if column_a != column_b:
                ax = df.plot(kind="line", x=column_a, y=column_b)
                plt.savefig(directory + '/'+column_a+'_' +
                            column_b+'.png', bbox_inches='tight')
                plt.close()


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


def initial_vorticity(frame, grid, dt_inv):
    cg = grid
    df = pd.DataFrame(frame[:, :, 0]).stack(dropna=False).rename_axis(
        ['y', 'x']).reset_index(name='flow_u')
    df_1 = pd.DataFrame(frame[:, :, 1]).stack(dropna=False).rename_axis(
        ['y', 'x']).reset_index(name='flow_v')
    df['flow_v'] = df_1['flow_v']
    print('done pivoting')
    df = latlon_converter(df, cg)
    print('done converting to lat lon')
    df = scaling_df_approx(df, cg, dt_inv)
    print('done scaling')
    _, omega = vorticity(df)
    print('done vorticity')
    print('omega shape final:')
    print(omega.shape)

    return omega


def initial_flow(frame, grid, dt_inv):
    mean=0
    sigma=0.1
    gaussianx = np.random.normal(mean, sigma, (frame.shape[0],frame.shape[1]))
    gaussiany = np.random.normal(mean, sigma, (frame.shape[0],frame.shape[1])) 
    frame[:,:,0]=frame[:,:,0]+gaussianx
    frame[:,:,1]=frame[:,:,1]+gaussiany
    #frame[:,:,1]-cv2.blur(frame[:,:,1],(3,3))
    #frame[:,:,0]-cv2.blur(frame[:,:,0],(3,3))
    cg = grid
    df = pd.DataFrame(frame[:, :, 0]).stack(dropna=False).rename_axis(
        ['y', 'x']).reset_index(name='flow_u')
    df_1 = pd.DataFrame(frame[:, :, 1]).stack(dropna=False).rename_axis(
        ['y', 'x']).reset_index(name='flow_v')
    df['flow_v'] = df_1['flow_v']
    print('done pivoting')
    df = latlon_converter(df, cg)
    print('done converting to lat lon')
    print(frame.shape)
    frame[:, :, 0], frame[:, :, 1] = scaling_df_approx_inv(df, cg, dt_inv)
   
    

    return frame


def dataframe_pivot(frame, var):
    df = pd.DataFrame(frame).stack().rename_axis(
        ['y', 'x']).reset_index(name=var.lower())
    return df


def scaling_df_approx00(df, grid, dt_inv):
    """coordinate transforms vels from angle/pixel to metric, approximately"""

    df['u_scaled_approx'] = scaling_u(
        df['lon'], df['lat'], df['flow_u'], grid, dt_inv)
    df['v_scaled_approx'] = scaling_v(
        df['lon'], df['lat'], df['flow_v'], grid, dt_inv)
    
    return df


def scaling_df_approx_inv(df, grid, dt_inv):
   """coordinate transforms vels from angle/pixel to metric, approximately"""

   df['flow_u'] = scaling_u_inv(df['lon'], df['lat'], df['flow_u'], grid, dt_inv)
   df['flow_v'] = scaling_v_inv( df['lon'], df['lat'], df['flow_v'], grid, dt_inv)
   flowx = df.pivot('y', 'x', 'flow_u').values
   flowy = df.pivot('y', 'x', 'flow_v').values
   print('flowx shape')
   print(df.shape)
   print(flowx.shape)

   return flowx, flowy


def vorticity(df):
    print('Calculating vorticity...')
    
    u_a = pd.pivot_table(df, values='u_scaled_approx',
                         index=["y"], columns=["x"], fill_value=0)
    v_a = pd.pivot_table(df, values='v_scaled_approx',
                         index=["y"], columns=["x"], fill_value=0)
    lon = pd.pivot_table(df, values='lon',
                         index=["y"], columns=["x"], fill_value=0)
    lat = pd.pivot_table(df, values='lat',
                         index=["y"], columns=["x"], fill_value=0)
    u_a = u_a.to_numpy()
    v_a = v_a.to_numpy()
    lon = lon.to_numpy()
    lat = lat.to_numpy()
    dx, dy = metpy.calc.lat_lon_grid_deltas(lon, lat)
    f = metpy.calc.coriolis_parameter(np.deg2rad(lat)).to(units('1/sec'))
    omega = metpy.calc.vorticity(u_a * units['m/s'],
                                 v_a * units['m/s'], dx, dy, dim_order='yx')

    omega = omega.magnitude
    omega = np.nan_to_num(omega)

    omega = cv2.blur(omega, (3, 3))

    print('omega shape:')
    print(omega.shape)
    df_u = pd.DataFrame(omega).stack().rename_axis(
        ['y', 'x']).reset_index(name='vorticity')
    df = df.merge(df_u, how='left')
    return df, omega


def scaling_df_approx(df, grid, dt_inv):

    
    
   
    lon = pd.pivot_table(df, values='lon',
                         index=["y"], columns=["x"], fill_value=0)
    lat = pd.pivot_table(df, values='lat',
                         index=["y"], columns=["x"], fill_value=0)

    lon = lon.to_numpy()
    lat = lat.to_numpy()
    dx, dy = metpy.calc.lat_lon_grid_deltas(lon, lat)

    dx = pd.DataFrame(dx.magnitude).stack().rename_axis(
        ['y', 'x']).reset_index(name='dx')
    dy = pd.DataFrame(dy.magnitude).stack().rename_axis(
        ['y', 'x']).reset_index(name='dy')
    df['u_scaled_approx']=dt_inv*dx['dx']*df['flow_u']
    df['v_scaled_approx']=dt_inv*dy['dy']*df['flow_v']

 
    return df


def scaling_u_inv(df_lon, df_lat, df_flow_u, grid, dt_inv):
    dtheta = grid
    drads = dtheta * math.pi / 180
    lat = df_lat*math.pi/90/2
    # dt_hr = 1
    # dt_s = 3600
    R = 6371000
    scaleConstant = dt_inv
    dx = R*abs(np.cos(lat))*drads
    scale = dx*scaleConstant
    return df_flow_u/scale


def scaling_v(df_lon, df_lat, df_flow_v, grid, dt_inv):
    """coordinate transform for v from pixel/angular to metric, approximate"""
    dtheta = grid*df_flow_v
    drads = dtheta * math.pi / 180
    R = 6371000
    scaleConstant = dt_inv
    dx = R*drads
    scale = dx*scaleConstant
    return scale


def scaling_v_inv(df_lon, df_lat, df_flow_v, grid, dt_inv):
    """coordinate transform for v from pixel/angular to metric, approximate"""
    dtheta = grid
    drads = dtheta * math.pi / 180
    R = 6371000
    scaleConstant = dt_inv
    dx = R*drads
    scale = dx*scaleConstant
    return df_flow_v/scale


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
