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
sns.set_context("talk")
#sns.set_context('poster')
pd.set_option('display.expand_frame_repr', False)
dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))

def plotter(df,values):
    piv = pd.pivot_table(df, values=values,
                         index=["lat"], columns=["lon"], fill_value=0)
    U= pd.pivot_table(df, values='u',
                         index=["lat"], columns=["lon"], fill_value=0)
    V= pd.pivot_table(df, values='v',
                         index=["lat"], columns=["lon"], fill_value=0)
   
    print(u.to_numpy())
    X=np.arange(-180,180,0.0625)
    Y=np.arange(-90,90,0.0625)
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.quiver(X,Y,U,V)
    pmap=matplotlib.cm.get_cmap('YlGnBu')
    pmap.set_bad(color='black')
    im = ax.imshow(piv, cmap=pmap,extent=[-180,180,-90,90],origin='lower')
    cbar=fig.colorbar(im, ax=ax)
   

    cbar.set_label('m/s')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(values)
    directory='../data/processed/poster'
    plt.savefig(directory + '/'+values+'.png', bbox_inches='tight', dpi=1000)



start_date=datetime.datetime(2006,7,1,11,0,0,0)
end_date=datetime.datetime(2006,7,1,12,0,0,0)
#df = aa.df_concatenator(dataframes_dict, start_date, end_date)
#print(df)

plotter(df, 'qv')