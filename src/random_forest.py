#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
"""
from viz import amv_analysis as aa
from viz import dataframe_calculators as dfc 
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


mpl.rcParams['figure.dpi']= 150
sns.set_context("paper")
#sns.set_context('poster')
pd.set_option('display.expand_frame_repr', False)
dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


start_date=datetime.datetime(2006,7,1,6,0,0,0)
end_date=datetime.datetime(2006,7,1,7,0,0,0)
df = aa.df_concatenator(dataframes_dict, start_date, end_date, False, True,False)
df=df.dropna()

regressor =RandomForestRegressor( n_estimators=100, random_state=0, n_jobs=-1)

X=df[['lat','u_scaled_approx','v_scaled_approx']]
Y=df[['u','v']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9)

print('fitting')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)



print('done!')

