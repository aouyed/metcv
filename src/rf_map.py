#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
"""
from joblib import dump, load
from viz import amv_analysis as aa
from viz import dataframe_calculators as dfc
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
from sklearn.model_selection import train_test_split

sns.set_style('white', {'legend.frameon': None})
# mpl.rcParams['figure.dpi'] = 150
# sns.set_context("paper")
sns.set_context('talk')
pd.set_option('display.expand_frame_repr', False)
dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


def shaded_plot(x, y, err, ax, co):
    ax.plot(x, y, color=co)
    ax.fill_between(x, y-err, y+err, alpha=0.6, color=co)


def plotter(df, values):
    grid = 10
    piv = pd.pivot_table(df, values=values,
                         index=["lat"], columns=["lon"], fill_value=float('NaN'))
    u = pd.pivot_table(df, values='u',
                       index=["lat"], columns=["lon"], fill_value=0)
    v = pd.pivot_table(df, values='v',
                       index=["lat"], columns=["lon"], fill_value=0)

    # print(u.to_numpy())
    U = u.to_numpy()
    V = v.to_numpy()
    factor = 0.0625/grid

    U = cv2.resize(U, None, fx=factor, fy=factor)
    V = cv2.resize(V, None, fx=factor, fy=factor)
    print(U.shape)
    print(V.shape)
    X = np.arange(-180, 180, grid)
    Y = np.arange(-90, 90, grid)
    print(len(X))
    print(len(Y))
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    ax.quiver(X, Y, U, V, scale=250)
    pmap = plt.cm.BuGn
    pmap.set_bad(color='black')
    im = ax.imshow(piv, cmap=pmap, extent=[-180, 180, -90, 90], origin='lower')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    cbar.set_label('g/kg')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title('Random Forest')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def scatter_plotter(X, values):
    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    ax.scatter(X['lon'], X['lat'], s=0.1)
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title('Training Data')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def line_plotter(X, X_2, X_3, values):
    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    sns.lineplot(X['speed'], X['speed_approx'], label='jpl', ax=ax)
    sns.lineplot(X_2['speed'], X_2['speed_approx'], label='physics', ax=ax)
    sns.lineplot(X_3['speed'], X_3['speed_approx'], label='vem', ax=ax)

    sns.lineplot(X['speed'], X['speed'], label='truth', ax=ax)
    # ax.legend()

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("ground truth [m/s]")
    ax.set_ylabel("AMV [m/s]")
    ax.set_title('Wind Speeds')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    sns.lineplot(X['speed'], X['speed_approx_std'], label='jpl', ax=ax)
    sns.lineplot(X_2['speed'], X_2['speed_approx_std'], label='physics', ax=ax)
    sns.lineplot(X_3['speed'], X_3['speed_approx_std'], label='vem', ax=ax)

    # ax.legend()

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("ground truth [m/s]")
    ax.set_ylabel("stdev [m/s]")
    ax.set_title('weighted standard deviations')
    directory = '../data/processed/density_plots'
    plt.savefig('stdev.png', bbox_inches='tight', dpi=300)


def line_plotter_1(X, X_2, X_3, values):
    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    sns.lineplot(X['speed'], X['speed_error'], label='jpl', ax=ax)
    sns.lineplot(X_2['speed'], X_2['speed_error'], label='physics', ax=ax)
    sns.lineplot(X_3['speed'], X_3['speed_error'], label='vem', ax=ax)

    #sns.lineplot(X['speed'], X['speed'], label='truth', ax=ax)
    # ax.legend()

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("ground truth [m/s]")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title('RMSVD')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def line_plotter_0(df, values):
    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    sns.lineplot(df['hours'], df['jpl'], label='jpl',
                 linestyle='--', marker='o', ax=ax)
    sns.lineplot(df['hours'], df['df'], label='vem',
                 linestyle='--', marker='o', ax=ax)
    sns.lineplot(df['hours'], df['rf'], label='physics',
                 linestyle='--', marker='o', ax=ax)

    ax.legend(frameon=None)

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("hour")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title('Results')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


# dump(regressor, 'rf.joblib')
regressor = load('rf18z.joblib')

hours = [0, 6, 12, 18]
rf = [1.7122288824810241, 1.6554113125780026,
      1.614328436366025, 1.7043883044378514]
deepf = [3.0814844441258487, 2.8653066164477585,
         2.7922954295197595, 2.9260425156666905]
jpl = [3.7132896180961383, 3.412857290296869,
       3.363429985554871, 3.440039715662562]


d = {'hours': hours, 'df': deepf, 'jpl': jpl, 'rf': rf}
df_results = pd.DataFrame(data=d)
print(df_results)

start_date = datetime.datetime(2006, 7, 1, 6, 0, 0, 0)
end_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)
df = aa.df_concatenator(dataframes_dict, start_date,
                        end_date, False, True, False)

df = df.dropna()

X = df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx']]
Y = df[['u', 'v']]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.99, random_state=1)

print('predicting...')
y_pred = regressor.predict(X)


# df['u_scaled_approx'] = y_pred[:, 0]

# df['v_scaled_approx'] = y_pred[:, 1]

df_jpl = aa.df_concatenator(dataframes_dict, start_date,
                            end_date, True, True, False)
df_jpl = df_jpl.dropna()
# df['speed_approx'] = np.sqrt(df['u_scaled_approx']**2+df['v_scaled_approx']**2)

xlistv = np.arange(df['speed'].min(), df['speed'].max(), 1)
print('averaging...')
df_mean = dfc.plot_average(
    deltax=1, df=df_jpl, xlist=xlistv, varx='speed', vary='speed_approx')
df_mean_e = dfc.plot_average(
    deltax=1, df=df_jpl, xlist=xlistv, varx='speed', vary='speed_error')
df_mean_vem = dfc.plot_average(
    deltax=1, df=df, xlist=xlistv, varx='speed', vary='speed_approx')
df_mean_vem_e = dfc.plot_average(
    deltax=1, df=df, xlist=xlistv, varx='speed', vary='speed_error')
df['speed_approx'] = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
df['speed_error'] = np.sqrt(
    (y_pred[:, 0]-df['u'])**2 + (y_pred[:, 1]-df['v'])**2)
df_mean_rf = dfc.plot_average(
    deltax=1, df=df, xlist=xlistv, varx='speed', vary='speed_approx')
df_mean_rf_e = dfc.plot_average(
    deltax=1, df=df, xlist=xlistv, varx='speed', vary='speed_error')
print('avg error')
print(df['speed_error'].mean())

df['qv'] = 1000*df['qv']
# plotter(df, 'qv')

df_mean_e['speed_error'] = np.sqrt(df_mean_e['speed_error'])
df_mean_vem_e['speed_error'] = np.sqrt(df_mean_vem_e['speed_error'])
df_mean_rf_e['speed_error'] = np.sqrt(df_mean_rf_e['speed_error'])

print('plotting...')
line_plotter(df_mean, df_mean_rf, df_mean_vem, 'speed')
line_plotter_1(df_mean_e, df_mean_rf_e, df_mean_vem_e, 'speed_error')

line_plotter_0(df_results, 'results')
