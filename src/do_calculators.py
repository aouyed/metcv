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


def cross_correlation(parameters):
    print('initializing cross correlation...')
    kwargs = vars(parameters)
    cc.cross_correlation_amv(**kwargs)
    print('finished cross correlation.')


def optical_flow(parameters):
    print('initializing optical flow...')
    kwargs = vars(parameters)
    ofa.optical_flow(**kwargs)
    print('finished optical flow.')


def coarsener(parameters):
    print('initializing coarsener...')
    kwargs = vars(parameters)
    c.coarsener(**kwargs)
    print('finished coarsener.')


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


def path(parameters):
    kwargs = vars(parameters)
    path = file_string(**kwargs)
    return(path)


def file_string(var, winsizes, poly_n, levels, iterations, cutoff, grid, **kwargs):
    path = (var+'_w'+str(winsizes)+'_p'+str(poly_n)+'_l'
            + str(levels)+'_i'+str(iterations)+'_c'+str(cutoff)+'_g'+str(grid))
    return path


def df_sumnmary(df, coarse):
    today = date.today()
    df_path = '../data/interim/dataframes/'+str(today)
    plot_path = '../data/processed/summary_plots/'+str(today)
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    df.to_pickle(df_path+'/c'+str(coarse)+'.pkl')
    print(df[["corr_speed", "corr_u", "corr_v", "ratio_count", "mean", "std"]])
   # dfc.line_plotter(df[['cutoff', 'corr_speed',
   #                      'mean_speed_error', 'initial_count', 'ratio_count']], plot_path)
