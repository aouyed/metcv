#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
,"""
import pdb
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
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
from global_land_mask import globe


def error_calc(df, f, name, category, rmse):
    error_uj = (df['umeanh'] - df['u_scaled_approx'])
    error_vj = (df['vmeanh'] - df['v_scaled_approx'])
    speed_errorj = (error_uj**2+error_vj**2)*df['cos_weight']
   # print('rmsvd for ' + name)
    f.write('rmsvd for '+name+'\n')
    rmsvd = np.sqrt(speed_errorj.sum()/df['cos_weight'].sum())
   # print(rmsvd)
    category.append(name)
    rmse.append(rmsvd)
    f.write(str(rmsvd)+'\n')


def ml(X, Y, name, f, df, extra, alg, category, rmse, tsize, only_land):
    # change df0z to df for current timestep

    X_train0, X_test0, y_train0, y_test0 = train_test_split(
        df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx', 'utrack', 'land']], df[['umeanh', 'vmeanh', 'utrack', 'land']], test_size=tsize, random_state=1)
    if only_land:
        X_train0 = X_train0[X_train0.land == True]
        X_train = X_train0[['lat', 'lon',
                            'u_scaled_approx', 'v_scaled_approx']]
        y_train0 = y_train0[y_train0.land == True]
    else:
        X_train = X_train0[['lat', 'lon',
                            'u_scaled_approx', 'v_scaled_approx', 'land']]

    y_train = y_train0[['umeanh', 'vmeanh']]

    if alg is 'rf':
        regressor = RandomForestRegressor(
            n_estimators=100, random_state=0, n_jobs=-1)

    else:
        polynomial_features = PolynomialFeatures(degree=5)
        X_train = polynomial_features.fit_transform(X_train)
        regressor = LinearRegression()

    # print('fitting')
    regressor.fit(X_train, y_train)
   # erase the two lines below the comment for current timestep
   # X_test0 = df[['lat', 'lon', 'u_scaled_approx',
   #               'v_scaled_approx', 'utrack']]
   # y_test0 = df[['umeanh', 'vmeanh', 'utrack']]

    X_test0['cos_weight'] = np.cos(X_test0['lat']/180*np.pi)
    if extra:
        X_test0 = X_test0[X_test0['utrack'].isna()]
        y_test0 = y_test0[y_test0['utrack'].isna()]
    else:
        X_test0 = X_test0.dropna()
        y_test0 = y_test0.dropna()
    i

    if alg is 'poly':
        if only_land:
            X_test = polynomial_features.fit_transform(
                X_test0[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx']])
        else:
            X_test = polynomial_features.fit_transform(
                X_test0[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx', 'land']])
    else:
        if only_land:
            X_test = X_test0[['lat', 'lon',
                              'u_scaled_approx', 'v_scaled_approx']]
        else:
            X_test = X_test0[['lat', 'lon',
                              'u_scaled_approx', 'v_scaled_approx', 'land']]

    y_pred = regressor.predict(X_test)

    error_u = (y_test0['umeanh'] - y_pred[:, 0])
    error_v = (y_test0['vmeanh'] - y_pred[:, 1])

    speed_error = (error_u**2+error_v**2)*X_test0['cos_weight']
    # print("rmsvd for "+alg+"_"+name)

    f.write("rmsvd for" + alg+"_"+name+"\n")
    rmsvd = np.sqrt(speed_error.sum()/X_test0['cos_weight'].sum())
    # print(rmsvd)
    f.write(str(rmsvd)+'\n')
    category.append(alg)
    rmse.append(rmsvd)


def latitude_selector(df, dft, lowlat, uplat, extra, category, rmse, latlon, extras, test_size, test_sizes, only_land):
    dfm = df[(df.lat) <= uplat]
    dfm = df[(df.lat) >= lowlat]

#    df0zm = df0z[(df0z.lat) <= uplat]
 #   df0zm = df0z[(df0z.lat) >= lowlat]

    dftm = dft[(dft.lat) <= uplat]
    dftm = dft[(dft.lat) >= lowlat]

    X = dfm[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx']]
    Y = dfm[['umeanh', 'vmeanh']]

    if lowlat < 0:
        lowlat = str(abs(lowlat)) + '°S'
    else:
        lowlat = str(lowlat) + '°N'

    if uplat < 0:
        uplat = str(abs(uplat)) + '°S'
    else:
        uplat = str(uplat) + '°N'

    latlon.append(str(str(lowlat)+',' + str(uplat)))
    extras.append(extra)
    test_sizes.append(test_size)
    ml(X, Y, 'uv', f,  dfm.copy(),
       extra, 'poly', category, rmse, test_size, only_land)
    latlon.append(str(str(lowlat)+',' + str(uplat)))
    extras.append(extra)
    test_sizes.append(test_size)
    ml(X, Y, 'uv', f,  dfm.copy(),
       extra, 'rf', category, rmse, test_size, only_land)
    if extra:
        dfm = dfm[dfm['utrack'].isna()]
    else:
        dfm = dfm.dropna()
    latlon.append(str(str(lowlat)+',' + str(uplat)))
    extras.append(extra)
    test_sizes.append(test_size)
    error_calc(dfm, f, "df", category, rmse)
    latlon.append(str(str(lowlat)+',' + str(uplat)))
    extras.append(extra)
    test_sizes.append(test_size)
    error_calc(dftm, f, 'jpl', category, rmse)


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


# df0z = pd.read_pickle("df_0z.pkl")
# df0z = df0z.dropna(subset=['qv'])
# df0z = df0z.dropna(subset=['umeanh'])

df = df.dropna(subset=['qv'])
df = df.dropna(subset=['umeanh'])
df['land'] = globe.is_land(df.lat, df.lon)
# print(df['land'])


dft = aa.df_concatenator(dataframes_dict, start_date,
                         end_date, True, True, False)
f = open("errors.txt", "w+")

dft = dft.dropna()
dft['land'] = globe.is_land(dft.lat, dft.lon)
#
category = []
rmse = []
latlon = []
extras = []
test_sizes = []

test_size = 0.99

only_land = True

latdowns = [-30, 30, 60, -60, -90]
latups = [30, 60, 90, -30, -60]
extral = [True, False]
test_sizel = [0.99, 0.999]
print('process data...')
for i, latdown in enumerate(tqdm(latdowns)):
    for extra in extral:
        for test_size in test_sizel:
            latitude_selector(df.copy(), dft.copy(), latdown, latups[i],
                              extra, category, rmse, latlon, extras, test_size, test_sizes, only_land)

d = {'latlon': latlon, 'extra': extras, 'categories': category,
     'test_size': test_sizes, 'rmse': rmse}
df_results = pd.DataFrame(data=d)
print('done!')
print(df_results)
df_results.to_pickle("df_results.pkl")
