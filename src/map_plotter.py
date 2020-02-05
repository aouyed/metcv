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
sns.set_context("paper")
#sns.set_context('poster')
pd.set_option('display.expand_frame_repr', False)
dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))

def shaded_plot(x,y,err,ax, co):
    ax.plot(x,y, color=co)
    ax.fill_between(x, y-err, y+err, alpha=0.6, color=co)

def plotter(df,values):
    grid=10
    piv = pd.pivot_table(df, values=values,
                         index=["lat"], columns=["lon"], fill_value=float('NaN'))
    u= pd.pivot_table(df, values='u',
                         index=["lat"], columns=["lon"], fill_value=0)
    v= pd.pivot_table(df, values='v',
                         index=["lat"], columns=["lon"], fill_value=0)
   
    #print(u.to_numpy())
    U=u.to_numpy()
    V=v.to_numpy()
    factor=0.0625/grid

    U = cv2.resize(U,None,fx=factor,fy=factor)
    V = cv2.resize(V,None,fx=factor,fy=factor)
    print(U.shape)
    print(V.shape)
    X=np.arange(-180,180 -grid,grid)
    Y=np.arange(-90,90-grid,grid)
    print(len(X))
    print(len(Y))
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
            
    ax.quiver(X,Y,U,V, scale=250)
    pmap=plt.cm.BuGn
    pmap.set_bad(color='black')
    im = ax.imshow(piv, cmap=pmap,extent=[-180,180,-90,90],origin='lower',vmin=-1, vmax=0)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    cbar=fig.colorbar(im, ax=ax,fraction=0.025, pad=0.04)
   

    cbar.set_label('kg/g')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title('grund truth')
    directory='../data/processed/density_plots'
    plt.savefig(directory + '/'+values+'.png', bbox_inches='tight', dpi=300)



start_date=datetime.datetime(2006,7,1,6,0,0,0)
end_date=datetime.datetime(2006,7,1,7,0,0,0)
df = aa.df_concatenator(dataframes_dict, start_date, end_date, False, True)
df=df.dropna()
#df_jpl = aa.df_concatenator(dataframes_dict, start_date, end_date, True, True)


#d#f['delta_error']=-(df['speed_error']-df_jpl['speed_error'])
#df#['qv']=1000*df['qv']

#df_jpl['qv']=1000*df_jpl['qv']
df['vorticity']=1e4*df['vorticity']
plotter(df,'vorticity')