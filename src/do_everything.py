#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:59:16 2019

@author: amirouyed
"""
from datetime import datetime
import do_calculators as dc
import pandas as pd
from data import make_dataset_geos5 as gd
from features import build_features as qvd

class Parameters:
    def __init__(self, **kwargs):
        prop_defaults = {
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
            "target_box": 10, 
            'sub_pixel': False,
            "path": "_",
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

def downloader(parameters):
    print('initializing  downloader...')
    parameters.var = 'U'
    kwargs = vars(parameters)
    gd.downloader(**kwargs)

    parameters.var = 'V'
    kwargs = vars(parameters)
    gd.downloader(**kwargs)

    parameters.var = 'QV'
    kwargs = vars(parameters)
    gd.downloader(**kwargs)

    parameters.var = 'AIRDENS'
    kwargs = vars(parameters)
    gd.downloader(**kwargs)

    parameters.var = 'QVDENS'
    qvd.builder(parameters.var)
    
    if parameters.grid >0.0625:
        dc.coarsener(parameters)

    print('finished downloader.')

def processor(parameters, parameters_process):
    """ iteratres through the hyperparameters"""
    print('initializing processor...')
    cutoffs = parameters.cutoffs
    df = pd.DataFrame()
  
  
    size_path = dc.path(parameters)
    parameters.path = size_path
    if parameters_process.do_cross_correlation:
        dc.cross_correlation(parameters)
    if parameters_process.do_optical_flow:
        dc.optical_flow(parameters)
    if parameters_process.do_builder:
        dc.builder(parameters)
    for cutoff in cutoffs:
        parameters.cutoff = cutoff
        size_path = dc.path(parameters)
        parameters.path = size_path
        if parameters_process.do_analysis:     
            df_unit = dc.analysis(parameters)
            df = dc.df_parameters(df, df_unit, parameters)

        #mm.frame_maker(var, size_path)
    dc.df_sumnmary(df, parameters.coarse)
    print('finished processor.')



