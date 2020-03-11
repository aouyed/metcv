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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load


def error_calc_rf(X, regressor):

    X_test = X[X.lat <= 30]
    X_test = X_test[X_test.lat >= -30]
    X_test, y_pred = X_test_init_rf(X_test, regressor)

    error_uj = X_test['umeanh'] - y_pred[:, 0]
    error_vj = X_test['vmeanh']-y_pred[:, 1]
    X_test['cos_weight'] = np.cos(X_test['lat']/180*np.pi)
    speed_errorj = (error_uj**2+error_vj**2)*X_test['cos_weight']
    rmsvd = np.sqrt(speed_errorj.sum()/X_test['cos_weight'].sum())
    print('tropics for rf')
    print(rmsvd)


def rf(X, Y, name, f, df):
    # regressor = RandomForestRegressor(
     #   n_estimators=100, random_state=0, n_jobs=-1)

    polynomial_features = PolynomialFeatures(degree=5)

    # model = LinearRegression()
    # model.fit(x_poly, y)
    regressor = LinearRegression()
    # X = df[[ 'lat', 'lon', 'u_scaled_approx', 'v_scaled_approx']]
    # X = df[['lat', 'lon', 'qv']]
    # Y = df[['u', 'v']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.99, random_state=1)
    X_trainp = polynomial_features.fit_transform(X_train)
    X_testp = polynomial_features.fit_transform(X_test)
    print('train size:')
    print(X_train.shape)
    print('test size')
    print(X_test.shape)
    print('fitting')
    regressor.fit(X_trainp, y_train)
    y_pred = regressor.predict(X_testp)

    hr = '0z'
    dump(regressor, 'rf_'+name+'.joblib')
    dump(X_train, 'xtr_'+name+'.joblib')
    dump(X_test, 'xte_'+name+'.joblib')
    dump(y_test, 'yte_'+name+'.joblib')
    dump(y_train, 'ytr_'+name+'.joblib')

    X_test['cos_weight'] = np.cos(X_test['lat']/180*np.pi)
    error_u = (y_test['umeanh'] - y_pred[:, 0])
    error_v = (y_test['vmeanh'] - y_pred[:, 1])

    # df = df.dropna(subset=['qv'])

# df['qv'] = 1000*df['qv']
# plotter(df, 'qv')

    df = df[df['utrack'].isna()]
    y_pred = regressor.predict(
        polynomial_features.fit_transform(df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx']]))

    error_u = (df['umeanh'] - y_pred[:, 0])
    error_v = (df['vmeanh'] - y_pred[:, 1])

    speed_error = (error_u**2+error_v**2)*df['cos_weight']
    print("rmsvd for rf_"+name)
    f.write("rmsvd for rf_"+name+"\n")
    rmsvd = np.sqrt(speed_error.sum()/df['cos_weight'].sum())
    print(rmsvd)
    f.write(str(rmsvd)+'\n')


def diff_qv(df):
    qv = df.pivot('y', 'x', 'qv').values
    qv = np.nan_to_num(qv)
    diff_qv = np.gradient(qv)
    print(np.mean(diff_qv))
    diff_qv = np.array(diff_qv)
    diff_qv = np.nan_to_num(diff_qv)
    df_1 = pd.DataFrame(diff_qv[0, :, :]).stack().rename_axis(
        ['y', 'x']).reset_index(name='dqv_dy')
    df = df.reset_index()

    df['dqv_dy'] = df_1['dqv_dy']
    df_1 = pd.DataFrame(diff_qv[1, :, :]).stack().rename_axis(
        ['y', 'x']).reset_index(name='dqv_dx')
    df['dqv_dx'] = df_1['dqv_dx']
    return df


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

df = df.dropna(subset=['qv'])
df = df.dropna(subset=['umeanh'])
df = df[df.lat <= 60]
df = df[df.lat >= 30]
dft = aa.df_concatenator(dataframes_dict, start_date,
                         end_date, True, True, False)
dft = dft.dropna()
dft = dft[dft.lat <= 60]
dft = dft[dft.lat >= 30]
f = open("errors.txt", "w+")

X = df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx']]
Y = df[['umeanh', 'vmeanh']]
rf(X, Y, 'uv', f, df.copy())

df = df[df['utrack'].isna()]
error_uj = (df['umeanh'] - df['u_scaled_approx'])
error_vj = (df['vmeanh'] - df['v_scaled_approx'])
speed_errorj = (error_uj**2+error_vj**2)*df['cos_weight']
print('rmsvd for deepflow and whole dataset')
f.write('rmsvd for deepflow and whole dataset\n')
rmsvd = np.sqrt(speed_errorj.sum()/df['cos_weight'].sum())
print(rmsvd)
f.write(str(rmsvd)+'\n')
error_ujt = (dft['umeanh'] - dft['u_scaled_approx'])
error_vjt = (dft['vmeanh'] - dft['v_scaled_approx'])
speed_errorjt = (error_ujt**2+error_vjt**2)*dft['cos_weight']
print('rmsvd for jpl')
f.write('rmsvd for jpl\n')
rmsvd = np.sqrt(speed_errorjt.sum()/dft['cos_weight'].sum())
print(rmsvd)
f.write(str(rmsvd)+'\n')
f.close()
print('done!')
