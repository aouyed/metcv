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
from joblib import dump, load


mpl.rcParams['figure.dpi'] = 150
sns.set_context("paper")
# sns.set_context('poster')
pd.set_option('display.expand_frame_repr', False)



dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


start_date = datetime.datetime(2006, 7, 1, 6, 0, 0, 0)
end_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)
df = aa.df_concatenator(dataframes_dict, start_date,
                        end_date, False, True, False)
df = df.dropna()
print(df)
regressor = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)

X = df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx']]
Y = df[['u', 'v']]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.99, random_state=1)

print('fitting')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

hr = '0z'
dump(regressor, 'rf.joblib')
dump(X_train, 'xtr.joblib')
dump(X_test, 'xte.joblib')
dump(y_test, 'yte.joblib')
dump(y_train, 'ytr.joblib')


dft = aa.df_concatenator(dataframes_dict, start_date,
                         end_date, True, True, False)
dft = dft.dropna()


f = open("errors.txt", "w+")


X_test['cos_weight'] = np.cos(X_test['lat']/180*np.pi)
error_u = (y_test['u'] - y_pred[:, 0])
error_v = (y_test['v'] - y_pred[:, 1])

speed_error = (error_u**2+error_v**2)*X_test['cos_weight']
print("rmsvd for rf")
f.write("rmsvd for rf\n")
rmsvd = np.sqrt(speed_error.sum()/X_test['cos_weight'].sum())
print(rmsvd)
f.write(str(rmsvd)+'\n')


error_uj = (df['u'] - df['u_scaled_approx'])
error_vj = (df['v'] - df['v_scaled_approx'])
speed_errorj = (error_uj**2+error_vj**2)*df['cos_weight']
print('rmsvd for deepflow and whole dataset')
f.write('rmsvd for deepflow and whole dataset\n')
rmsvd = np.sqrt(speed_errorj.sum()/df['cos_weight'].sum())
print(rmsvd)
f.write(str(rmsvd)+'\n')
error_ujt = (dft['u'] - dft['u_scaled_approx'])
error_vjt = (dft['v'] - dft['v_scaled_approx'])
speed_errorjt = (error_ujt**2+error_vjt**2)*dft['cos_weight']
print('rmsvd for jpl')
f.write('rmsvd for jpl\n')
rmsvd = np.sqrt(speed_errorjt.sum()/dft['cos_weight'].sum())
print(rmsvd)
f.write(str(rmsvd)+'\n')
f.close()
print('done!')
