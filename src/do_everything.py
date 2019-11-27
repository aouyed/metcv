#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:59:16 2019

@author: amirouyed
"""
import matplotlib as mpl
from data import make_dataset_geos5 as gd
from computer_vision import optical_flow as of
from viz import amv_analysis as aa
from datetime import datetime
from features import build_features as qvd
from viz import moviemaker as mm

def iterator(var, winsizes,levelses,poly_ns,iterationses,cutoffs):
    for cutoff in cutoffs:
        for poly_n in poly_ns:
            for iterations in iterationses:
                for levels in levelses:
                    for winsize in winsizes:
                        size_path=(var+'_w'+str(winsize)+'_p'+str(poly_n)+'_l'
                                   +str(levels)+'_i'+str(iterations)+'_c'+str(cutoff))
                        print(size_path)
                        #mm.frame_maker(var, size_path)
                        of.optical_flow_calculator(d0,var, pyr_scale, levels, 
                                                   winsize, iterations, poly_n, poly_sigma)
                        aa.dataframe_builder(d1,var,dtheta)
                        aa.data_analysis(d0_sample,d1_sample,var,size_path,cutoff)


d0 = datetime(2006, 7, 1,0,0,0,0)
d1 = datetime(2006, 7, 1,4,0,0,0)
d0_sample = datetime(2006, 7, 1,0,0,0,0)
d1_sample = datetime(2006, 7, 1,4,0,0,0)
dtheta=0.5
#winsizes=[1,5,10,100]
#levelses=[1,5,10,30]
#poly_ns=[1,5,10,15]
#iterationses=[3]
#cutoffs=[0.3,0.1,-1]
winsizes=[10]
levelses=[5]
poly_ns=[2]
iterationses=[3]
cutoffs=[-1]
poly_sigma=1.2
pyr_scale=0.5

#gd.downloader(d0,d1,'QV','qv')
#gd.downloader(d0,d1,'U','u')
#gd.downloader(d0,d1,'V','v')
#gd.downloader(d0,d1,'AIRDENS','airdens')


#iterator('AIRDENS',winsizes,levelses,poly_ns,iterationses,cutoffs)

#qvd.builder('qvdens')
iterator('QVDENS',winsizes,levelses,poly_ns,iterationses,cutoffs)

print('Done_final')