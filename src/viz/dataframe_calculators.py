#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:56:34 2019

@author: aouyed
"""

import glob
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import numpy as np
import os 
import matplotlib.pyplot as plt
import pickle 
import pandas as pd
import metpy.calc as mpcalc

def daterange(start_date, end_date):
    date_list = []
    delta = timedelta(hours=1)
    while start_date < end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list

def heatmap_plotter(df,date,directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    df=df.loc[[date]]
    for column in df:
        if column != 'x' and column != 'y' and column != 'lat' and column != 'lon' and column !='datetime':
            heatmapper(df,column,date,directory)

def heatmapper(df,values,date,directory):
    piv = pd.pivot_table(df, values=values,index=["lat"], columns=["lon"], fill_value=0)
    ax = sns.heatmap(piv)
    ax.set_title(values)
    plt.tight_layout()
    plt.savefig(directory +'/'+values+'.png',bbox_inches='tight')
    plt.close()
    #plt.show()

def plotter(df,directory,date):
    if not os.path.exists(directory):
        os.makedirs(directory)
    df=df.loc[[date]]
    for column_a in df:
        for column_b in df:
            if(column_a != column_b):
                ax=df.plot(kind="scatter",x=column_a, y=column_b)
                plt.savefig(directory +'/'+column_a+'_'+column_b+'.png',bbox_inches='tight')
                plt.close()

    
def dataframe_quantum(file,date,dictionary_dict):
               
        frame=np.load(file)
        
        df=pd.DataFrame(frame[:,:,0]).stack().rename_axis(['y', 'x']).reset_index(name='flow_u')
        df_1=pd.DataFrame(frame[:,:,1]).stack().rename_axis(['y', 'x']).reset_index(name='flow_v')  
        df['flow_v']=df_1['flow_v']
        df['datetime'] = pd.Timestamp(date)
        
        for state_var in dictionary_dict:
            state_files=dictionary_dict[state_var]
            frame=np.load(state_files[date])        
            df_1=pd.DataFrame(frame).stack().rename_axis(['y', 'x']).reset_index(name=state_var.lower())
            df=df.merge(df_1,how='left')  
        return df
    
def scaling_df(df):
    df['u_scaled_approx']=df.apply(lambda x: scaling_lon(x.lon,x.lat,x.flow_u),axis=1)
    df['v_scaled_approx']=df.apply(lambda x: scaling_lat(x.lon,x.lat,x.flow_v),axis=1)
    df.to_pickle('dataframes/scales.pkl')
    return df

def scaling_df_approx(df):
    df['u_scaled_approx']=df.apply(lambda x: scaling_lon_approx(x.lon,x.lat,x.flow_u),axis=1)
    df['v_scaled_approx']=df.apply(lambda x: scaling_lat_approx(x.lon,x.lat,x.flow_v),axis=1)
    return df

def scaling_df_approx_airdens(df):
    df['flow_u_scaled_airdens_approx']=df.apply(lambda x: scaling_lon_approx(x.lon,x.lat,x.flow_u_airdens),axis=1)
    df['flow_v_scaled_airdens_approx']=df.apply(lambda x: scaling_lat_approx(x.lon,x.lat,x.flow_v_airdens),axis=1)
    #df.to_pickle('dataframes/scales_approx_whole.pkl')
    return df
    
    

def error_calculator(df):
    df["error_u"]=df['u']-df['flow_u']
    df["error_v"]=df['v']-df['flow_v']         
    df["error_u_norm"]=df["error_u"]/df['u']
    df["error_v_norm"]=df["error_v"]/df['v']
    
    return df

def scaling_lon(lon,lat,dpixel):
    dtheta=0.5*dpixel
    dt_hr=1
    dt_s=3600
    scaleConstant=(dt_hr/dt_s)
    lons=np.array([lon,lon+dtheta])
    lats=np.array([lat,lat])
    dx,dy=mpcalc.lat_lon_grid_deltas(lons,lats)
    dx=dx.magnitude
    scale=dx[0][0]*scaleConstant
    return(scale)
    
def scaling_lon_approx(lon,lat,dpixel):
    dtheta=0.5*dpixel
    drads=dtheta * np.pi / 180
    lat=lat*np.pi/90/2
    dt_hr=1
    dt_s=3600
    R=6371000
    scaleConstant=(dt_hr/dt_s)
    dx=R*abs(np.cos(lat))*drads
    scale=dx*scaleConstant
    return(scale)
    
def scaling_lat_approx(lon,lat,dpixel):
    dtheta=0.5*dpixel
    drads=dtheta * np.pi / 180
    dt_hr=1
    dt_s=3600
    R=6371000
    scaleConstant=(dt_hr/dt_s)
    dx=R*drads
    scale=dx*scaleConstant
    return(scale)
    
def scaling_lat(lon,lat,dpixel):
    dtheta=0.5*dpixel
    dt_hr=1
    dt_s=3600
    scaleConstant=(dt_hr/dt_s)
    lons=np.array([lon,lon])
    lats=np.array([lat,lat+dtheta])
    dx,dy=mpcalc.lat_lon_grid_deltas(lons,lats)
    dy=dy.magnitude
    scale=dy[0][0]*scaleConstant
    return(scale)
    
    
def latlon_converter(df,dtheta):
    df['lat']=df['y']*dtheta -90
    df['lon']=df['x']*dtheta -180
    return(df)