#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:22:39 2019

@author: aouyed
"""

import pandas as dataframe
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import dataframe_calculators as dfc
import seaborn as sns 
import os
        

#def plot_flow():
#        
#        plt.savefig(directory+'/pcolor'+str( df_0.plot(kind="scatter",x='lon',y='u_scaling_approx',title=str(date))
#        plt.savefig(directory+'/'+str(date)+'_'+'scaled_flow_u'+'.png')
#        plt.close()
#        df_0[df_0.y==0].plot(kind="scatter",x='lon',y='u_scaling_approx',title=str(date))
#        plt.savefig(directory+'/'+str(date)+'_'+'scaled_flow_u_0'+'.png')
#        plt.close()
#        df_0[df_0.y==1].plot(kind="scatter",x='lon',y='u_scaling_approx',title=str(date))
#        plt.savefig(directory+'/'+str(date)+'_'+'scaled_flow_u_1'+'.png')
#        plt.close()
#        df_0[df_0.y==2].plot(kind="scatter",x='lon',y='u_scaling_approx',title=str(date))
#        plt.savefig(directory+'/'+str(date)+'_'+'scaled_flow_u_2'+'.png')
#        plt.close()
#        df_0[df_0.x==0].plot(kind="line",x='lat',y='u_scaling_approx',title=str(date))
#        plt.savefig(directory+'/'+str(date)+'_'+'scaled_flow_u_lat'+'.png')
#        plt.close()
#        data = df_0.pivot(index='y', columns='x', values='flow_u')
#        sns.heatmap(data)
#        plt.savefig(directory+'/pcolor'+str(date)+'_'+'flow_u'+'.png')
#        plt.close()
#        data = df_0.pivot(index='y', columns='x', values='u_scaling_approx')
#        sns.heatmap(data)date)+'_'+'scaled_flow_u'+'.png')
#        plt.close()


def plotter_artificial(df_0,date,directory):
        print(directory)
        dfc.plotter(df_0[['lat','lon','flow_u','flow_v',
                          'u_scaling_approx','v_scaling_approx']],directory+'/'+str(date)+'_')


       

def df_test(dtheta,size_path):
    path='dictionaries/optical_flow_artificial.pkl'
    
    artificial_dict=pickle.load(open(path,'rb'))
    directory='artificial_figures_'+size_path
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    random_date=list(artificial_dict.keys())[0]
    frame=np.load(artificial_dict[random_date])
    artificial_dict.pop(random_date)
    df=pd.DataFrame(frame[:,:,0]).stack().rename_axis(['y', 'x']).reset_index(name='flow_u')
    df_1=pd.DataFrame(frame[:,:,1]).stack().rename_axis(['y', 'x']).reset_index(name='flow_v')  
    df['flow_v']=df_1['flow_v']
    df['datetime'] = pd.Timestamp(random_date)
    
    for date in artificial_dict:
        frame=np.load(artificial_dict[date])
        df_0=pd.DataFrame(frame[:,:,0]).stack().rename_axis(['y', 'x']).reset_index(name='flow_u')
        df_1=pd.DataFrame(frame[:,:,1]).stack().rename_axis(['y', 'x']).reset_index(name='flow_v')  
        df_0['flow_v']=df_1['flow_v']
        df_0['lat']=df_0['y']*dtheta -90
        df_0['lon']=df_0['x']*dtheta -180
        df_0=dfc.scaling_df_approx(df_0)
        plotter_artificial(df_0,date,directory)
        df_0['datetime'] = pd.Timestamp(date)
        df=pd.concat((df,df_0))
