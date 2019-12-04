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
from features import build_features as qvd
import viz
from viz import moviemaker as mm
import pandas as pd
import os


class Parameters:
    def __init__(self, start_date, end_date, var, pyr_scale, levels,
                 winsize, iterations, poly_n, poly_sigma, cutoff):
        self.start_date = start_date
        self.end_date = end_date
        self.var = var
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.cutoff = cutoff


def df_parameters(df, df_unit, parameters):
    for parameter, value in parameters.__dict__.items():
        df_unit[parameter] = value
    if df.empty:
        df = df_unit
    else:
        df = pd.concat([df, df_unit])
    return df


def iterator(var, winsizes, levelses, poly_ns, iterationses, cutoffs):
    """ iteratres through the hyperparameters"""
    df = pd.DataFrame()
    for cutoff in cutoffs:
        for poly_n in poly_ns:
            for iterations in iterationses:
                for levels in levelses:
                    for winsize in winsizes:
                        parameters = Parameters(d0, d1, var, pyr_scale, levels,
                                                winsize, iterations, poly_n, poly_sigma, cutoff)
                        size_path = (var+'_w'+str(winsize)+'_p'+str(poly_n)+'_l'
                                     + str(levels)+'_i'+str(iterations)+'_c'+str(cutoff)+'_g'+str(grid))
                        print(size_path)
                        #mm.frame_maker(var, size_path)
                        #of.optical_flow_calculator(d0, var, pyr_scale, levels,
                        #                          winsize, iterations, poly_n,poly_sigma)
                        #aa.dataframe_builder(d1, var, grid,dt)
                        d_sample=datetime(2006, 7, 1, 0, 30, 0, 0)
                        df_unit = aa.data_analysis(d_sample, d_sample,
                                                   var, size_path, cutoff)

                        df = df_parameters(df, df_unit, parameters)
                        print(df.loc['speed_error'])
                        
    df_path = '../data/interim/dataframes'
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    today = date.today()
    #df.to_pickle(df_path+'/'+str(today)+'.pkl')


d0 = datetime(2006, 7, 1, 0, 0, 0, 0)
d1 = datetime(2006, 7, 1, 1, 0, 0, 0)
grid = 0.0625
#seconds
dt=1800
winsizes = [10]
levelses = [5]
poly_ns = [2]
iterationses = [3]
cutoffs = [-1]
poly_sigma = 1.2
pyr_scale = 0.5

level=65
coarse=False
#gd.downloader(d0,d1,'QV','qv',level, coarse)
#gd.downloader(d0,d1,'U','u',level, coarse)
#gd.downloader(d0,d1,'V','v',level, coarse)
#gd.downloader(d0,d1,'AIRDENS','airdens',level, coarse)
qvd.builder('qvdens')
#gd.data_diagnostic('U',d0)
#gd.pressure_diagnostic('PL',d0)

iterator('QVDENS', winsizes, levelses, poly_ns, iterationses, cutoffs)

print('Done_final')
