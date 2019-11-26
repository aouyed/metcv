#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:07:59 2019

@author: aouyed
"""

import numpy as np
from datetime import datetime
import pickle
from scipy.ndimage.interpolation import shift
from datetime import datetime
from datetime import timedelta
import dataframe_calculators as dfc




def artificial_generator(d0,d1,var,directory,vel,shape):
    delta = d1 - d0
    date_list= dfc.daterange(d0, d1)
    test_image=np.full(shape,255)
    #test_image= np.mgrid[0:255:730j]
    #test_image[:]=255
    
    
    file_paths={}
    
    
    for date in date_list:
            test_image=shift(test_image, vel, cval=0)
            #print(test_image)
            #test_image=np.vstack((test_image,test_image))
            file_path=str(directory+'/'+str(date)+".npy")
            np.save(file_path,test_image)
            file_paths[date]=str(directory+'/'+str(date)+".npy")
    
    f = open('dictionaries/'+ var+'.pkl',"wb")
    pickle.dump(file_paths,f)