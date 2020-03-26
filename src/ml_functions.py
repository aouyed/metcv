
import pdb
import matplotlib.pyplot as plt
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
from global_land_mask import globe
from sklearn.utils import resample
import extra_data_plotter as edp

R = 6373.0


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


def ml_fitter(name, f, df,  alg, rmse, tsize, only_land, lowlat, uplat, exp_filter):

    X_train0, X_test0, y_train0, y_test0 = train_test_split(df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx', 'utrack', 'land', 'sample_weight', 'umeanh', 'vmeanh', 'distance']], df[[
        'umeanh', 'vmeanh', 'utrack', 'land', 'lat']], test_size=tsize, random_state=1)

    deltax = 10
    distances = np.arange(0, 1800, deltax)

    if exp_filter:
        X = pd.DataFrame()
        print('sampling based on distance...')
        for distance in tqdm(distances):
            X_sample = X_train0[(X_train0.distance >= distance) & (
                X_train0.distance <= X_train0.distance + deltax)]
            n_int = int(round(1e5 *
                              np.exp(-(distance+0.5*deltax)/18000)))
            X_sample = resample(X_train0, replace=False,
                                n_samples=n_int, random_state=1)
            if X.empty:
                X = X_sample
            else:
                X = pd.concat([X, X_sample])
                X_train0 = X

    exp_distance = np.exp(X_train0.distance/18000)
    scale_noise_u = abs(X_train0['umeanh']*exp_distance)
    scale_noise_v = abs(X_train0['vmeanh']*exp_distance)
    X_train0['umeanh'] = X_train0.umeanh + \
        np.random.normal(scale=scale_noise_u)
    X_train0['vmeanh'] = X_train0.vmeanh + \
        np.random.normal(scale=scale_noise_v)
    y_train0 = X_train0[['umeanh', 'vmeanh']]

    sample_weight = X_train0['sample_weight']
    X_train = X_train0[['lat', 'lon',
                        'u_scaled_approx', 'v_scaled_approx', 'land']]

    y_train = y_train0[['umeanh', 'vmeanh']]

    regressor = RandomForestRegressor(
        n_estimators=100, random_state=0, n_jobs=-1)

    # print('fitting')
    regressor.fit(X_train, y_train)
    return regressor, X_test0, y_test0


def ml_predictor(name, f, alg, category,   rmse, tsize, lowlat, uplat, regressor, X_test0, y_test0):
     # change df0z to df for current timestep

    X_test0['cos_weight'] = np.cos(X_test0['lat']/180*np.pi)
    X_test0 = X_test0.dropna()
    y_test0 = y_test0.dropna()

    X_test0 = X_test0[(X_test0.lat >= lowlat) & (X_test0.lat <= uplat)]
    y_test0 = y_test0[(y_test0.lat >= lowlat) & (y_test0.lat <= uplat)]

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


def latitude_selector(f, df, dft, lowlat, uplat,  category, rmse, latlon, test_size, test_sizes, only_land, exp_filter, exp_list, regressor, X_test0, y_test0):
    dfm = df[(df.lat) <= uplat]
    dfm = df[(df.lat) >= lowlat]

    dftm = dft[(dft.lat) <= uplat]
    dftm = dft[(dft.lat) >= lowlat]
    lowlat0 = lowlat
    uplat0 = uplat

    if lowlat < 0:
        lowlat = str(abs(lowlat)) + '°S'
    else:
        lowlat = str(lowlat) + '°N'

    if uplat < 0:
        uplat = str(abs(uplat)) + '°S'
    else:
        uplat = str(uplat) + '°N'
    latlon.append(str(str(lowlat)+',' + str(uplat)))
    test_sizes.append(test_size)
    exp_list.append(exp_filter)

    ml_predictor('uv', f, 'rf', category, rmse, test_size, lowlat0,
                 uplat0, regressor, X_test0, y_test0)
    dfm = dfm.dropna()
    latlon.append(str(str(lowlat)+',' + str(uplat)))
    test_sizes.append(test_size)
    exp_list.append(exp_filter)

    error_calc(dfm, f, "df", category, rmse)
    latlon.append(str(str(lowlat)+',' + str(uplat)))
    test_sizes.append(test_size)
    exp_list.append(exp_filter)

    error_calc(dftm, f, 'jpl', category, rmse)
