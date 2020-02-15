#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:20:28 2020

@author: aouyed
"""

from viz import amv_analysis as aa
from viz import dataframe_calculators as dfc 
import datetime
import pickle 

dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


start_date=datetime.datetime(2006,7,1,5,0,0,0)
end_date=datetime.datetime(2006,7,1,7,0,0,0)
df = aa.df_concatenator(dataframes_dict, start_date, end_date, False, True)
df=df.dropna()