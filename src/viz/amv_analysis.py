#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:27:02 2019

@author: amirouyed
"""

from viz import dataframe_calculators as dfc
import glob
from datetime import datetime
import numpy as np
import os 
import matplotlib.pyplot as plt
import pickle 
import pandas as pd
import metpy.calc as mpcalc
import seaborn as sns 

    
def dataframe_builder(end_date,var,dtheta):  
    dictionary_paths=glob.glob('../data/interim/dictionaries/*')
    dict_optical_paths=glob.glob('../data/interim/dictionaries_optical_flow/*')
    dictionary_dict={}
    dictionary_dict_optical={}
    print(dictionary_paths)
    
    for path in dictionary_paths:
        var_name=os.path.basename(path).split('.')[0]
        dictionary_dict[var_name]=pickle.load(open(path,'rb'))
        print(path)
        
    for path in dict_optical_paths:
        var_name=os.path.basename(path).split('.')[0]
        dictionary_dict_optical[var_name]=pickle.load(open(path,'rb'))
                 
    flow_files=dictionary_dict_optical[var]
    random_date=list(flow_files.keys())[0]
    df=dfc.dataframe_quantum(flow_files[random_date],random_date,dictionary_dict)
    df=dfc.latlon_converter(df,dtheta)
    df=dfc.scaling_df_approx(df)
    flow_files.pop(random_date)
    
    for date in flow_files:
        file=flow_files[date]
        df_quantum=dfc.dataframe_quantum(file,date,dictionary_dict)
        df_quantum=dfc.latlon_converter(df_quantum,dtheta)
        df_quantum=dfc.scaling_df_approx(df_quantum)
        df=pd.concat([df,df_quantum])
    
    df.set_index('datetime', inplace=True)
    df_path='../data/interim/dataframes'
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    df.to_pickle(df_path+'/'+var+'.pkl')


def df_printer(df,directory):
    df_prints_path='../../data/processed/df_prints'
    if not os.path.exists(df_prints_path):
        os.makedirs(df_prints_path)
    with open(df_prints_path+'/'+directory+'.txt', 'w') as f:
        print(df[['flow_v','v_scaled_approx','v','error_v_abs']].describe(),file=f)
        print(df[['flow_u','u_scaled_approx','u','error_u_abs']].describe(),file=f)
        print(df[['v_scaled_approx','v','u_scaled_approx','u']].corr(method ='pearson'),file=f)
        print(df[['speed','speed_error','speed_approx']].describe(),file=f)
        print(df[['speed','speed_error','speed_approx']].corr(method ='pearson'),file=f)

def absolute_df(df):
    df["error_u"]=df['u']-df['u_scaled_approx']
    df["error_v"]=df['v']-df['v_scaled_approx'] 
    df["error_u_abs"]=abs(df["error_u"])
    df["error_v_abs"]=abs(df["error_v"])
    df["u_abs"]=abs(df["u"])
    df["v_abs"]=abs(df["v"])
    df['speed']=np.sqrt(df['u']*df['u']+df['v']*df['v'])
    df['speed_approx']=np.sqrt(df['u_scaled_approx']*df['u_scaled_approx']+df['v_scaled_approx']*df['v_scaled_approx'])
    df['speed_error']=abs(df["speed"]-df['speed_approx'])

    return df
    
def data_analysis(start_date,end_date,var,directory,cutoff):
   
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.expand_frame_repr', False)
    df_path='../data/interim/dataframes/'+var+'.pkl'
    df_path=os.path.abspath(df_path)
    df = pd.read_pickle(df_path)     
    df=df[start_date:end_date]
    df=absolute_df(df)
    if(cutoff>0):
        df=df[df.speed_error<= cutoff*df['speed'].max()]

    df_printer(df,directory)
    
    
    scatter_directory='../data/processed/scatter_'+directory
    dfc.plotter(df[['flow_v','v_scaled_approx','v','error_v']],scatter_directory,end_date)
    dfc.plotter(df[['flow_u','u_scaled_approx','u','error_u']],scatter_directory,end_date)
    dfc.plotter(df[['speed','speed_approx','speed_error']],scatter_directory,end_date)

    heatmap_directory='../data/processed/heatmaps_'+directory
    dfc.heatmap_plotter(df,end_date,heatmap_directory)
    ax=df.loc[[end_date]].plot(kind="scatter",x='speed', y='speed_error')
    #plt.savefig(scatter_directory +'/'+'derek_speed.png',bbox_inches='tight')
    plt.close()

    print('Done!')