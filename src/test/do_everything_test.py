#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:59:16 2019

@author: amirouyed
"""
import matplotlib as mpl
import geos5downloader_opendap_xr as gd
import optical_flow as of
import amv_analysis as aa
import scale_matrix_calculator as smc
from datetime import datetime
import artificial_test as at
import artificial_df as adf
#mpl.use('Agg')

pyr_scale = 0.5 
dtheta=0.5
levels = 3
winsize = 300
iterations = 3 
poly_n = 5
poly_sigma = 1.2
d0 = datetime(2006, 7, 1,0,0,0,0)
d1 = datetime(2006, 7, 1,4,0,0,0)
shape=(360,720)
var='artificial'
directory='artificial'
vel=3


polys=[1]
winsizes=[100]
vels=[3]
levelses=[3,10,20]

for level in levelses:
    for vel in vels:
        for poly_n in polys:
            for winsize in winsizes:
                size_path=('w'+str(winsize)+'_p'+str(poly_n)+'_v'+str(vel)+'_l'+str(level))
                at.artificial_generator(d0,d1,var,directory,vel,shape)
                of.optical_flow_calculator(d0,'artificial', pyr_scale, levels, 
                                   winsize, iterations, poly_n, poly_sigma)
                adf.df_test(dtheta,size_path)
