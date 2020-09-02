#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:29:30 2019

@author: amirouyed
"""
from datetime import date

from computer_vision import optical_flow as of
from computer_vision import optical_flow_all as ofa
from computer_vision import cross_correlation as cc
from computer_vision import coarsener as c
from viz import amv_analysis as aa
from viz import dataframe_calculators as dfc
import pandas as pd
import os


def df_parameters(df, df_unit, parameters):
    for parameter, value in parameters.__dict__.items():
        if not isinstance(value, list):
            df_unit[parameter] = value
    if df.empty:
        df = df_unit
    else:
        df = pd.concat([df, df_unit])
    return df


def optical_flow(parameters):
    print('initializing optical flow...')
    kwargs = vars(parameters)
    ofa.optical_flow(**kwargs)
    print('finished optical flow.')


def builder(parameters):
    print('initializing builder...')
    kwargs = vars(parameters)
    aa.dataframe_builder(**kwargs)
    print('finished builder.')


def analysis(parameters):
    print('initializing analysis...')
    kwargs = vars(parameters)
    df_unit = aa.data_analysis(**kwargs)
    print('finished analysis.')
    return df_unit


def df_sumnmary(df):
    print(df[["corr_speed", "corr_u", "corr_v", "ratio_count", "mean", "std"]])
