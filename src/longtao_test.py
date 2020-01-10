#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
"""
from viz import amv_analysis as aa
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import cartopy.crs as ccrs
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import cv2
from copy import copy

sns.set_context("talk")
#sns.set_context('poster')
pd.set_option('display.expand_frame_repr', False)
dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))
sns.set_style('white')
directory='../data/processed/poster'


def summary_plot(df_path,df_path_fine,label1,label2):
    df = pd.read_pickle(df_path)
    dff = pd.read_pickle(df_path_fine)
    fig, ax = plt.subplots()
    sns.lineplot(x=df['ratio_count'],y=df['corr_speed'], ax=ax, label=label1)
    sns.lineplot(x=dff['ratio_count'],y=dff['corr_speed'], ax=ax,label=label2)
    ax.set_xlabel('Yield')
    ax.set_ylabel('correlation coefficient')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(directory + '/corr.png', bbox_inches='tight', dpi=2000)

    fig, ax = plt.subplots()
    sns.lineplot(x=df['ratio_count'],y=df['mean_speed_error'], ax=ax, label=label1)
    sns.lineplot(x=dff['ratio_count'],y=dff['mean_speed_error'], ax=ax,label=label2)
    ax.set_xlabel('Yield')
    ax.set_ylabel('mean vector difference [m/s]')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(directory + '/mvd.png', bbox_inches='tight', dpi=2000)


def plotter(df,values, u_version,v_version, error,label, vmin, vmax, grid ,scale, title, cmap ):
    dfc=df.copy()
    dfc['qv']=1000*dfc['qv']
    dfc.loc[dfc.speed_error>error,u_version]=0
    dfc.loc[dfc.speed_error>error,v_version]=0
    dfc.loc[dfc.speed_error>error,values]=float('NaN')
    piv = pd.pivot_table(dfc, values=values,
                         index=["lat"], columns=["lon"], fill_value=None)
    u= pd.pivot_table(dfc, values=u_version,
                         index=["lat"], columns=["lon"], fill_value=0)
    v= pd.pivot_table(dfc, values=v_version,
                         index=["lat"], columns=["lon"], fill_value=0)
   
    U=u.to_numpy()
    V=v.to_numpy()
    factor=0.0625/grid
    U = cv2.resize(U,None,fx=factor,fy=factor)
    V = cv2.resize(V,None,fx=factor,fy=factor)

    X=np.arange(-180,180,grid)
    Y=np.arange(-90,90,grid)
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    palette = copy(plt.cm.PuBuGn)
    palette.set_bad('grey', 1.0)



    ax.quiver(X,Y,U,V,scale=scale)
    if vmin>=0:
        im = ax.imshow(piv, cmap=palette, extent=[-180,180,-90,90],origin='lower',vmin=vmin, vmax=vmax)
    else:   
        im = ax.imshow(piv, cmap=palette, extent=[-180,180,-90,90],origin='lower')

   
    cbar=fig.colorbar(im, ax=ax, orientation ='horizontal',fraction=0.1, pad=0.04)
   

    cbar.set_label(label)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(title)
    directory='../data/processed/poster'
    plt.savefig(directory + '/'+values+'_'+u_version+'.png', bbox_inches='tight', dpi=2000)



start_date=datetime.datetime(2006,7,1,5,0,0,0)
end_date=datetime.datetime(2006,7,1,7,0,0,0)
df = aa.df_concatenator(dataframes_dict, start_date, end_date, True, True)
#df['u']
print(df['u'])

#plotter(df, 'speed','u','v',np.inf, 'Wind speed [m/s]',0,40,10,250, 'Ground truth',sns.cm.rocket_r)
#plotter(df, 'speed_approx','u_scaled_approx','v_scaled_approx', np.inf,'Wind speed [m/s]',0,40,10,250, 'Atmospheric motion vectors',sns.cm.rocket_r)
#plotter(df, 'qv','u','v',np.inf, 'Water vapor [g/kg]',0,20,10,250, 'Ground truth','PuBuGn')
#plotter(df, 'qv','u_scaled_approx','v_scaled_approx',5, 'Water vapor [g/kg]',0,20,10,250, 'Atmospheric motion vectors','PuBuGn')


#df_path='../data/interim/dataframes/2020-01-03/dof_qv.pkl'
#df_path_fine='../data/interim/dataframes/2020-01-03/cc_qv.pkl'

#summary_plot(df_path,df_path_fine,'Dense optical flow','Cross correlation')