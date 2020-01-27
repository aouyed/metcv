#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
"""
from viz import amv_analysis as aa
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import cv2
sns.set_context("paper")
#sns.set_context('poster')
pd.set_option('display.expand_frame_repr', False)
dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))

def plotter(df,values):
    grid=10
    piv = pd.pivot_table(df, values=values,
                         index=["lat"], columns=["lon"], fill_value=0)
    u= pd.pivot_table(df, values='u_scaled_approx',
                         index=["lat"], columns=["lon"], fill_value=0)
    v= pd.pivot_table(df, values='v_scaled_approx',
                         index=["lat"], columns=["lon"], fill_value=0)
   
    #print(u.to_numpy())
    U=u.to_numpy()
    V=v.to_numpy()
    factor=0.0625/grid

    U = cv2.resize(U,None,fx=factor,fy=factor)
    V = cv2.resize(V,None,fx=factor,fy=factor)
    print(U.shape)
    print(V.shape)
    X=np.arange(-180,180-grid,grid)
    Y=np.arange(-90,90-grid,grid)
    print(len(X))
    print(len(Y))
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
            
    ax.quiver(X,Y,U,V, scale=500)
    im = ax.imshow(piv, cmap="BuGn",extent=[-180,180,-90,90],origin='lower', vmax=3, vmin=0)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    cbar=fig.colorbar(im, ax=ax)
   

    cbar.set_label('m/s')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(values)
    directory='../data/processed/density_plots'
    plt.savefig(directory + '/'+values+'.png', bbox_inches='tight', dpi=300)



start_date=datetime.datetime(2006,7,1,6,0,0,0)
end_date=datetime.datetime(2006,7,1,7,0,0,0)
df = aa.df_concatenator(dataframes_dict, start_date, end_date, False, True)
df_jpl = aa.df_concatenator(dataframes_dict, start_date, end_date, True, True)
df_f=df[df.lat<=-60]
df_f_jpl=df_jpl[df_jpl.lat<=-60]



df['delta_error']=df['speed_error']-df_jpl['speed_error']
df['delta_error_jpl']=-df['delta_error']

df=df.dropna()
df_f=df_f.dropna()
df_f_jpl=df_f_jpl.dropna()

print(df['speed_error'].mean())
print(df_jpl['speed_error'].mean())
print(df_f['speed_error'].mean())
print(df_f_jpl['speed_error'].mean())
print(df_f['speed_approx'].mean())
print(df_f_jpl['speed_approx'].mean())
plotter(df, 'delta_error')