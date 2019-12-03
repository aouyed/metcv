#!/usr/bin/env python3,
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:22:13 2019

@author: amirouyed
"""


import pickle 
import os
import numpy as np
import xarray as xr
from datetime import datetime
from datetime import timedelta

def daterange(start_date, end_date,dhour):
    date_list = []
    delta = timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


def pressure_diagnostic(var, start_date):
    url = u'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.0625_deg/inst/inst30mn_3d_PL_Nv'
    ds = xr.open_dataset(url,decode_times=True)
    date=start_date
    ds= ds.sel(time=date,method='nearest')
    
    for level in range(60,73):
        T = ds.sel(lev=level, lon=slice(-180,180),lat=slice(-90,90))
        T=T.get([var.lower()])            #print(T)
        T=T.to_array()
        T=np.squeeze(T)
        mean=np.mean(T.values)/100
        stdev=np.std(T.values)/100
        print(str(level)+' '+str(mean)+' '+str(stdev))
    


def downloader(start_date, end_date,opendap_var,directory,level,coarse):
    d0=start_date
    d1=end_date
    if coarse:    
        url = u'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.5000_deg/inst/inst01hr_3d_'+opendap_var+'_Cp'
        date_list= daterange(d0, d1,1)

    else:
        url = u'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.0625_deg/inst/inst30mn_3d_'+opendap_var+'_Nv'
        date_list= daterange(d0, d1,0.5)

            
    ds = xr.open_dataset(url,decode_times=True)
    file_paths={}
    for date in date_list:
            try:
                print(opendap_var)
                print(date)
                T= ds.sel(time=date,method='nearest')
                T = T.sel(lev=level, lon=slice(-180,180),lat=slice(-90,90))
    
                T=T.get([opendap_var.lower()])            #print(T)
                T=T.to_array()
                T=np.squeeze(T)
                print(T.shape)
                level = 71 
                X = np.arange(-130.0, -64.99, .5) # -65 is the last element
                Y = np.arange(25.0, 50.01, .5) # 50 is the last element
                file_path=str(directory+'/'+str(date)+".npy")
                np.save(file_path,T.values)
                file_paths.update({date:file_path})
            except Exception as e:
                print(type(e).__name__, e)
    dictionary_path='../data/interim/dictionaries'
    if not os.path.exists( dictionary_path):
        os.makedirs(dictionary_path)
    f = open(dictionary_path+'/'+ opendap_var+'.pkl',"wb")
    pickle.dump(file_paths,f)

