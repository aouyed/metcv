#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:56:34 2019

@author: aouyed
"""

import glob
import pickle
from datetime import timedelta
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import metpy.calc as mpcalc
from scipy.interpolate import interpn
import math
from tqdm import tqdm


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


def plot_average(deltax, df, directory, xlist, varx, vary):
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_mean = pd.DataFrame()
    df_unit = pd.DataFrame(data=[0], columns=[varx])
    print("calculating averages ...")
    for x in tqdm(xlist):
        df_a = df[df[varx] >= x]
        df_a = df_a[df_a[varx] <= x+deltax]

        df_unit[varx] = x
        df_unit[vary] = df_a[vary].mean()
        if df_mean.empty:
            df_mean = df_unit
        else:
            df_mean = pd.concat([df_mean, df_unit])
    print("preparing line plots...")
    df_mean.to_pickle(directory+'/df_mean_'+varx+'_'+vary+'.pkl')
    line_plotter(df_mean, directory)


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

def dataframe_pivot(frame, var):
    df= pd.DataFrame(frame).stack().rename_axis(
    ['y', 'x']).reset_index(name=var.lower())
    return df
    

def scaling_df(df):
    """coordinate transforms vels from angle/pixel to metric exactly"""

    df['u_scaled_approx'] = df.apply(
        lambda x: scaling_lon(x.lon, x.lat, x.flow_u), axis=1)
    df['v_scaled_approx'] = df.apply(
        lambda x: scaling_lat(x.lon, x.lat, x.flow_v), axis=1)
    df.to_pickle('dataframes/scales.pkl')
    return df


def scaling_df_approx(df, grid, dt_inv):
    """coordinate transforms vels from angle/pixel to metric, approximately"""
    df['u_scaled_approx'] = df.apply(
        lambda x: scaling_lon_approx(x.lon, x.lat, x.flow_u, grid, dt_inv), axis=1)
    df['v_scaled_approx'] = df.apply(
        lambda x: scaling_lat_approx(x.lon, x.lat, x.flow_v, grid, dt_inv), axis=1)
    return df


def error_calculator(df):
    """calculates error between ground truth and opticla flow values"""
    df["error_u"] = df['u']-df['flow_u']
    df["error_v"] = df['v']-df['flow_v']
    df["error_u_norm"] = df["error_u"]/df['u']
    df["error_v_norm"] = df["error_v"]/df['v']

    return df


def scaling_lon(lon, lat, dpixel):
    """coordinate transform for u from pixel/angular to metric, exact"""
    dtheta = 0.5*dpixel
    dt_hr = 1
    dt_s = 3600
    scaleConstant = (dt_hr/dt_s)
    lons = np.array([lon, lon+dtheta])
    lats = np.array([lat, lat])
    dx, _ = mpcalc.lat_lon_grid_deltas(lons, lats)
    dx = dx.magnitude
    scale = dx[0][0]*scaleConstant
    return scale


def scaling_lon_approx(lon, lat, dpixel, grid, dt_inv):
    """coordinate transform for u from pixel/angular to metric, approximate"""

    dtheta = grid*dpixel
    drads = dtheta * math.pi / 180
    lat = lat*math.pi/90/2
    # dt_hr = 1
    # dt_s = 3600
    R = 6371000
    scaleConstant = dt_inv
    dx = R*abs(math.cos(lat))*drads
    scale = dx*scaleConstant
    return scale


def scaling_lat_approx(lon, lat, dpixel, grid, dt_inv):
    """coordinate transform for v from pixel/angular to metric, approximate"""
    dtheta = grid*dpixel
    drads = dtheta * math.pi / 180
    # dt_hr = 1
    # dt_s = 3600
    R = 6371000
    scaleConstant = dt_inv
    dx = R*drads
    scale = dx*scaleConstant
    return scale


def scaling_lat(lon, lat, dpixel):
    """coordinate transform for v from pixel/angular to metric, exact"""

    dtheta = 0.5*dpixel
    dt_hr = 1
    dt_s = 3600
    scaleConstant = (dt_hr/dt_s)
    lons = np.array([lon, lon])
    lats = np.array([lat, lat+dtheta])
    _, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
    dy = dy.magnitude
    scale = dy[0][0]*scaleConstant
    return scale


def latlon_converter(df, dtheta):
    """coordinate transform for pixel to angular"""

    df['lat'] = df['y']*dtheta - 90
    df['lon'] = df['x']*dtheta - 180
    return df
