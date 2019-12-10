#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:59:16 2019

@author: amirouyed
"""
from datetime import datetime
from datetime import date
import matplotlib as mpl
from data import make_dataset_geos5 as gd
from computer_vision import optical_flow as of
from viz import amv_analysis as aa
from viz import dataframe_calculators as dfc
from features import build_features as qvd
import viz
from viz import moviemaker as mm
import pandas as pd
import os



class Parameters:
     def __init__(self, **kwargs):
         prop_defaults={
                 "start_date": datetime(2006, 7, 1, 0, 0, 0, 0),
                 "end_date": datetime(2006, 7, 1, 1, 0, 0, 0),
                 "var": "QVDENS",
                 "pyr_scale": 0.5,
                 "levels": 5,
                 "level": 65, 
                 "winsize": 10,
                 "iterations": 3,
                 "poly_n": 2,
                 "poly_sigma": 1.2,
                 "cutoff": 2.5,
                 "cutoffs": [2.5],
                 "grid": 0.0625,
                 "coarse": False,
                 "dt": 1800,
                 "path":"_"
                 }
         for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))


    


def df_parameters(df, df_unit, parameters):
    for parameter, value in parameters.__dict__.items():
        df_unit[parameter] = value
    if df.empty:
        df = df_unit
    else:
        df = pd.concat([df, df_unit])
    return df

def optical_flow(parameters):
    print('initializing optical flow...')
    kwargs=vars(parameters)
    of.optical_flow_calculator(**kwargs)
    print('finished optical flow.')

    
def builder(parameters):
    print('initializing builder...')
    kwargs=vars(parameters)
    aa.dataframe_builder(**kwargs)
    print('finished builder.')

    
def analysis(parameters):
    print('initializing analysis...')
    kwargs=vars(parameters)
    df_unit = aa.data_analysis(**kwargs)
    print('finished analysis.')
    return df_unit
 
def path(parameters):
    kwargs=vars(parameters)
    path=file_string(**kwargs)
    return( path)

def file_string(var, winsize, poly_n, levels, iterations, cutoff, grid,**kwargs):
     path = (var+'_w'+str(winsize)+'_p'+str(poly_n)+'_l'
                                     + str(levels)+'_i'+str(iterations)+'_c'+str(cutoff)+'_g'+str(grid))
     return path
 
def df_sumnmary(df,coarse):
    today = date.today()
    df_path = '../data/interim/dataframes/'+str(today)
    plot_path='../data/processed/summary_plots/'+str(today)
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    print(df)
    df.to_pickle(df_path+'/c'+str(coarse)+'.pkl')
    dfc.line_plotter(df[['cutoff', 'corr_speed',
                         'mean_speed_error','initial_count','ratio_count']], plot_path)
    

def processor(parameters,parameters_process):
    """ iteratres through the hyperparameters"""
    print('initializing processor...')
    cutoffs=parameters.cutoffs
    df = pd.DataFrame()
    for cutoff in cutoffs:
        parameters.cutoff=cutoff                       
        size_path = path(parameters)
        parameters.path=size_path
        print(size_path)
        if parameters_process.do_optical_flow:                 
            optical_flow(parameters)
        if parameters_process.do_builder:  
            builder(parameters)
        if parameters_process.do_analysis:
            df_unit=analysis(parameters) 
                      
        #mm.frame_maker(var, size_path)
        df = df_parameters(df, df_unit, parameters)
    df_sumnmary(df,parameters.coarse)
    print('finished processor.')


def downloader(parameters):
    print('initializing  downloader...')
    if(parameters.var=='QVDENS'):
        qvd.builder(parameters.var)
    else:
        kwargs=vars(parameters)
        gd.downloader(**kwargs)
    print('finished downloader.')



