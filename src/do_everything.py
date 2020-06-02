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
from data import jpl_loader as jl
from data import exp_loader as el
import shutil
import os


class Parameters:
    def __init__(self, **kwargs):
        prop_defaults = {
            "start_date": datetime(2006, 7, 1, 0, 0, 0, 0),
            "end_date": datetime(2006, 7, 1, 1, 0, 0, 0),
            "var": "QVDENS",
            "grid": 0.0625,
            "coarse": False,
            "dt": 1800,
            "target_box_x": [10],
            "target_box_y": [10],
            'sub_pixel': False,
            "path": "_",
            "jpl_loader": False,
            "track": False,
            "pressure": 850,
            "cores": 5,
            'deep_flow': True,
            'triplet': datetime(2006, 7, 1, 0, 0, 0, 0)
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))


def downloader_function(parameters):
    kwargs = vars(parameters)
    jl.loader(**kwargs)
    # el.loader(**kwargs)


def downloader(parameters):
    print('resetting dictionaries and arrays...')
    jl.resetter()

    print('initializing  downloader...')

    parameters.var = 'u'
    kwargs = vars(parameters)
    downloader_function(parameters)

    parameters.var = 'v'
    kwargs = vars(parameters)
    downloader_function(parameters)

    parameters.var = 'qv'
    kwargs = vars(parameters)
    downloader_function(parameters)

#    parameters.var = 'vtrack'
 #   kwargs = vars(parameters)
  #  downloader_function(parameters)

   # parameters.var = 'utrack'
   # kwargs = vars(parameters)
   # downloader_function(parameters)

    #parameters.var = 'umean'
    #kwargs = vars(parameters)
    # downloader_function(parameters)
    #parameters.var = 'vmean'
    #kwargs = vars(parameters)
    # downloader_function(parameters)

    parameters.var = 'umeanh'
    kwargs = vars(parameters)
    downloader_function(parameters)
    parameters.var = 'vmeanh'
    kwargs = vars(parameters)
    downloader_function(parameters)

    if parameters.grid > 0.0625:
        dc.coarsener(parameters)

    print('finished downloader.')


def processor(parameters, parameters_process):
    """ iteratres through the hyperparameters"""
    print('initializing processor...')
    df = pd.DataFrame()

    size_path = dc.path(parameters)
    parameters.path = size_path
    dc.optical_flow(parameters)
    dc.builder(parameters)
    size_path = dc.path(parameters)
    parameters.path = size_path
    df_unit = dc.analysis(parameters)
    df = dc.df_parameters(df, df_unit, parameters)
    dc.df_sumnmary(df, parameters.coarse)
    print('finished processor.')
