#!/usr/bin/env python3,
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:22:13 2019

@author: amirouyed
"""
import glob
import signal
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from datetime import timedelta
from viz import dataframe_calculators as dfc
import gc


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


def data_diagnostic(var, start_date):
    url = u'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.5000_deg/inst/inst01hr_3d_'+var+'_Cp'
    ds = xr.open_dataset(url, decode_times=True)
    date = start_date
    ds = ds.sel(time=date, method='nearest')
    level = 850
    T = ds.sel(lev=level, lon=slice(-180, 180), lat=slice(-90, 90))
    T = T.get([var.lower()])  # print(T)
    T = T.to_array()
    T = np.squeeze(T)
    lons = np.arange(-180, 180, 0.5)
    lats = np.arange(-90, 90, 0.5)
    zeros = []
    for lon in lons:
        for lat in lats:
            a = T.sel(lon=lon, lat=lat)
            a = a.values
            ilat = (90+lat)/0.5
            ilon = (180+lon)/0.5
            ilat = int(round(ilat))
            ilon = int(round(ilon))
            b = T.values[ilat, ilon]
            diff = a-b
            zeros.append(diff)
            if(a != b):
                print('incongruence ' + str(a) + ' ' + str(b))

    zeros = np.asarray(zeros)
    print('mean of zeroes: '+str(np.mean(zeros)))


def pressure_diagnostic(var, start_date, level):
    url = u'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.0625_deg/inst/inst30mn_3d_PL_Nv'

    ds = xr.open_dataset(url, decode_times=True)
    date = start_date
    ds = ds.sel(time=date, method='nearest')
    for level in range(60, 73):
        T = ds.sel(lev=level, lon=slice(-180, 180), lat=slice(-90, 90))
        T = T.get([var.lower()])  # print(T)
        T = T.to_array()
        T = np.squeeze(T)
        mean = np.mean(T.values)/100
        stdev = np.std(T.values)/100
        print(str(level)+' '+str(mean)+' '+str(stdev))

        
def pressure_interpolation(date, levelp):
    signal.signal(signal.SIGALRM, timeout_handler)

    df=pd.DataFrame()
    levels=[1, 72]
    levels=list(levels)
    for level in levels:
        print('calculating dataframe for level: '+ str(level))
        for var in tqdm(('pl','qv','u','v')):        
            filenames=glob.glob("../data/raw/ganymede/"+var+'/*')
            ds = xr.open_dataset(filenames[0])
           # print('downloading frame for var: ' + str(var))
            frame = ds.sel(lev=level)
            frame = frame.get(var.upper())
            frame=frame.values
            frame=np.squeeze(frame)
            #print('downloaded frame for var:'+str(var))
                    #frame = frame.to_array()
                    #frame = np.squeeze(pl_frame)
            name=var+'_l'+str(level)
            df_u=dfc.dataframe_pivot(frame,name)
            if df.empty:
                df=df_u
            else:
                df=df.merge(df_u, how='left')


    print(df)
    import pdb; pdb.set_trace()


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


def downloader(start_date, end_date, var, level, coarse, dt,  **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    d0 = start_date
    d1 = end_date
    directory_path = '../data/raw/'+var.lower()
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if coarse:
        url = u'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.5000_deg/inst/inst01hr_3d_'+var+'_Cp'
        date_list = daterange(d0, d1, 1)

    else:
        url = u'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.0625_deg/inst/inst30mn_3d_'+var+'_Nv'
        date_list = daterange(d0, d1, (dt/3600))

    print("Downloading dataset from url: " + url)

    ds = xr.open_dataset(url, decode_times=True)
    file_paths = {}
    for date in date_list:
        signal.alarm(15)

        while True:
            try:
                print('Downloading data for variable ' +
                      var + ' for date: ' + str(date))
                T = ds.sel(time=date, method='nearest')
                T = T.sel(lev=level, method='nearest')
                # T = T.sel(lev=level, lon=slice(-180,:),lat=slice(-90,:))
                T = T.get([var.lower()])
                T = T.to_array()
                T = np.squeeze(T)
                print('shape of downloaded array: ' + str(T.shape))
                file_path = str(directory_path+'/'+str(date)+".npy")
                np.save(file_path, T.values)
                file_paths[date] = file_path
                gc.collect()
            except TimeoutException:
                gc.collect()
                print('retrying download.')
                signal.alarm(15)
                continue  # continue the for loop if function A takes more than 5 second
            else:
                gc.collect()
                break
        # Reset the alarm
    signal.alarm(0)
    dictionary_path = '../data/interim/dictionaries/vars'
    if not os.path.exists(dictionary_path):
        os.makedirs(dictionary_path)
    f = open(dictionary_path+'/' + var+'.pkl', "wb")
    pickle.dump(file_paths, f)
