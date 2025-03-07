#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:59:16 2019

@author: amirouyed
"""
from datetime import datetime
import do_calculators as dc
import pandas as pd
from data import jpl_loader as jl
import shutil
import os


class Parameters:
    def __init__(self, **kwargs):
        prop_defaults = {
            "var": "qv",
            # "grid": 0.0625,
            "grid": 0.0267,
            "dt": 1800,
            "pressure": 850,
            "cores": 5,
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

   # parameters.var = 'u'
    kwargs = vars(parameters)
    downloader_function(parameters)

    print('finished downloader.')


def processor(parameters):
    """ iteratres through the hyperparameters"""
    print('initializing processor...')
    df = pd.DataFrame()
    dc.optical_flow(parameters)
    dc.builder(parameters)
    df_unit = dc.analysis(parameters)
    df = dc.df_parameters(df, df_unit, parameters)
    dc.df_sumnmary(df)
    print('finished processor.')
