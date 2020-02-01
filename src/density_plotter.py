#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
"""
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
mpl.rcParams['figure.dpi']= 150
sns.set_context("talk")
#sns.set_context('poster')
pd.set_option('display.expand_frame_repr', False)
dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))

start_date=datetime.datetime(2006,7,1,6,0,0,0)
end_date=datetime.datetime(2006,7,1,7,0,0,0)
df = aa.df_concatenator(dataframes_dict, start_date, end_date, False, True)
df_jpl = aa.df_concatenator(dataframes_dict, start_date, end_date, True, True)
df=df.dropna()
df_jpl=df_jpl.dropna()



df['delta_error']=df['speed_error']-df_jpl['speed_error']
df['delta_error_b']=df['speed_error']
df.loc[df['delta_error'] < 0, 'delta_error_b'] = 1
df.loc[df['delta_error'] >= 0, 'delta_error_b'] = 0


df['delta_error_jpl']=-df['delta_error']

df=df.dropna()
df_f=df_f.dropna()
#df_f_jpl=df_f_jpl.dropna()

print(df['speed_error'].mean())
print(df_jpl['speed_error'].mean())
#plotter(df, 'delta_error')
deltax = 1
xlist = np.arange(df['speed'].min(),df['speed'].max(), deltax)
df_mean=dfc.plot_average(deltax, df, xlist, 'speed', 'speed_approx')
df_mean_jpl=dfc.plot_average(deltax, df_jpl, xlist, 'speed', 'speed_approx')
fig, ax = plt.subplots()
x=df_mean['speed']
y=df_mean['speed_approx']
err=df_mean['speed_approx_std']
sns.lineplot(x,y, ax=ax, label='vem')
x=df_mean_jpl['speed']
y=df_mean_jpl['speed_approx']
err=df_mean_jpl['speed_approx_std']
sns.lineplot(x,y,ax=ax,label='jpl')
sns.lineplot(x,x,ax=ax, label='ground truth')
ax.set_xlabel('ground truth speed')
ax.set_ylabel('mean AMV speed')



###
fig, ax = plt.subplots()
err=df_mean['speed_approx_std']
sns.lineplot(x,err, ax=ax,label='vem')
err=df_mean_jpl['speed_approx_std']
sns.lineplot(x,err, ax=ax, label='jpl')
ax.set_xlabel('ground truth speed')
ax.set_ylabel('standard deviation')
###
fig, ax = plt.subplots()
x=df_mean['speed']
y=df_mean['speed_approx']
err=df_mean['speed_approx_std']
sns.lineplot(x,y, ax=ax, label='vem')
x=df_mean_jpl['speed']
y=df_mean_jpl['speed_approx']
err=df_mean_jpl['speed_approx_std']
sns.lineplot(x,y,ax=ax,label='jpl')
sns.lineplot(x,x,ax=ax)
ax.set_xlim(0,15)
ax.set_ylim(0,15)
ax.set_xlabel('ground truth speed')
ax.set_ylabel('AMV speed')
###
df_mean=dfc.plot_average(deltax, df, xlist, 'speed', 'speed_error')
df_mean_jpl=dfc.plot_average(deltax, df_jpl, xlist, 'speed', 'speed_error')
fig, ax = plt.subplots()
x=df_mean['speed']
y=df_mean['speed_error']
sns.lineplot(x,y, ax=ax,label='vem')
x=df_mean_jpl['speed']
y=df_mean_jpl['speed_error']
sns.lineplot(x,y, ax=ax, label='jpl')
ax.set_xlabel('ground truth speed')
ax.set_ylabel('Mean Vector Difference')

fig, ax = plt.subplots()
x=df_mean['speed']
y=df_mean['speed_error']
sns.lineplot(x,y, ax=ax,label='vem')
x=df_mean_jpl['speed']
y=df_mean_jpl['speed_error']
sns.lineplot(x,y, ax=ax, label='jpl')
ax.set_xlim(0,15)
ax.set_ylim(0,4)
ax.set_xlabel('ground truth speed')
ax.set_ylabel('Mean Vector Difference')


deltax = 5
xlist = np.arange(df['lat'].min(),df['lat'].max(), deltax)
df_mean=dfc.plot_average(deltax, df, xlist, 'lat', 'speed_approx')
df_mean_jpl=dfc.plot_average(deltax, df_jpl, xlist, 'lat', 'speed_approx')
df_mean_t=dfc.plot_average(deltax, df_jpl, xlist, 'lat', 'speed')

fig, ax = plt.subplots()
x=df_mean['lat']
y=df_mean['speed_approx']
err=df_mean['speed_approx_std']
sns.lineplot(x,y, ax=ax, label='vem')
x=df_mean_jpl['lat']
y=df_mean_jpl['speed_approx']
sns.lineplot(x,y,ax=ax,label='jpl')
y=df_mean_t['speed']
sns.lineplot(x,y,ax=ax,label='ground truth')
ax.set_xlabel('Latitude')
ax.set_ylabel('mean speed')

fig, ax = plt.subplots()
x=df_mean['lat']
y=df_mean['speed_approx']
err=df_mean['speed_approx_std']
sns.lineplot(x,err, ax=ax, label='vem')
x=df_mean_jpl['lat']
y=df_mean_jpl['speed_approx']
err=df_mean_jpl['speed_approx_std']
sns.lineplot(x,err,ax=ax,label='jpl')

ax.set_xlabel('latitude')
ax.set_ylabel('AMV standard deviation')

deltax = 5
xlist = np.arange(df['lat'].min(),df['lat'].max(), deltax)
df_mean=dfc.plot_average(deltax, df, xlist, 'lat', 'speed_error')
df_mean_jpl=dfc.plot_average(deltax, df_jpl, xlist, 'lat', 'speed_error')

fig, ax = plt.subplots()
x=df_mean['lat']
y=df_mean['speed_error']
sns.lineplot(x,y, ax=ax, label='vem')
x=df_mean_jpl['lat']
y=df_mean_jpl['speed_error']
sns.lineplot(x,y,ax=ax,label='jpl')
ax.set_xlabel('latitude')
ax.set_ylabel('mean vector difference')

#ax.set_xlim(0,15)
#ax.set_ylim(0,4)