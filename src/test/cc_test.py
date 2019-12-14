#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:49:14 2019

@author: amirouyed
"""
from skimage import data
from skimage.feature.register_translation import _upsampled_dft
import numpy as np
from scipy.ndimage.interpolation import shift
from computer_vision import cross_correlation as cc
import pandas as pd
from skimage.feature import register_translation
from scipy.ndimage.interpolation import shift

def artificial_generator(vel):
    
# The shift corresponds to the pixel offset relative to the reference image
    image = data.camera()
    offset_image = shift(image, vel, cval=0)
    print(f"Known offset (y, x): {vel}")
    vel, error, diffphase = register_translation(image, offset_image)
    print(f"Detected pixel offset (y, x): {vel}")
    return (image, offset_image)



prvs_image, next_image=artificial_generator(vel = 13.0)
next_image=next_image.real
print(np.mean(next_image))
flow=cc.amv_calculator(prvs_image, next_image,(256,256))
#df=pd.DataFrame(flow[:,:,0]).stack().rename_axis(['y', 'x']).reset_index(name='flow_u')
#df_1=pd.DataFrame(flow[:,:,1]).stack().rename_axis(['y', 'x']).reset_index(name='flow_v')  
#df['flow_v']=df_1['flow_v']
#df.plot(kind="line", x='y', y='flow_v')
#df.plot(kind="line", x='y', y='flow_u')
#df.plot(kind="line", x='x', y='flow_v')
#df.plot(kind="line", x='x', y='flow_u')
#print(next_image)
#print(np.mean(flow))