#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:44:19 2020

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


#y_pred = regressor.predict(X_test)
# print(y_pred.shape)
from joblib import dump, load

#dump(regressor, 'rf.joblib')
regressor = load('rf0z.joblib')

dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


start_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)
end_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)
df = aa.df_concatenator(dataframes_dict, start_date,
                        end_date, False, True, False)
df = df.dropna()


#regressor = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
X = df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx']]
Y = df[['u', 'v']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9)


print("training size")
print(X_train.shape)
print("test size")
print(X_test.shape)

print('predicting...')
#regressor.fit(X_train, y_train)
#y_pred = regressor.predict(X_train)


# print('done!')


y_pred = regressor.predict(X_test)

dft = aa.df_concatenator(dataframes_dict, start_date,
                         end_date, True, True, False)
dft = dft.dropna()

error_u = y_test['u'] - y_pred[:, 0]
error_v = y_test['v'] - y_pred[:, 1]

speed_error = np.sqrt(error_u**2+error_v**2)
print("mvd for rf")
print(speed_error.mean())

error_ut = y_test['u'] - X_test['u_scaled_approx']
error_vt = y_test['v'] - X_test['v_scaled_approx']

speed_errort = np.sqrt(error_ut**2+error_vt**2)
print('mvd for df')
print(speed_errort.mean())


error_uj = df['u'] - df['u_scaled_approx']
error_vj = df['v'] - df['v_scaled_approx']
speed_errorj = np.sqrt(error_uj**2+error_vj**2)
print('mvd for deepflow and whole dataset')
print(speed_errorj.mean())

error_ujt = dft['u'] - dft['u_scaled_approx']
error_vjt = dft['v'] - dft['v_scaled_approx']
speed_errorjt = np.sqrt(error_ujt**2+error_vjt**2)
print('mvd for jpl')
print(speed_errorjt.mean())
