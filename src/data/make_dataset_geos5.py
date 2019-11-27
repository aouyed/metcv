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

def daterange(start_date, end_date):
    date_list = []
    delta = timedelta(hours=1)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list




def downloader(start_date, end_date,opendap_var,directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    url = u'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.5000_deg/inst/inst01hr_3d_'+opendap_var+'_Cp'
    ds = xr.open_dataset(url,decode_times=True)
    d0 = datetime(2006, 7, 1,0,0,0,0)
    
    d1 = datetime(2006, 7, 2,0,0,0,0)
    d0=start_date
    d1=end_date
    file_paths={}
    
    delta = d1 - d0
    date_list= daterange(d0, d1)
    for date in date_list:
            try:
                print(opendap_var)
                print(date)
                T= ds.sel(time=date,method='nearest')
                T = T.sel(lev=850, lon=slice(-180,180),lat=slice(-90,90))
    
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

