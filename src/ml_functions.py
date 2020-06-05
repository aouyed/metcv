
import seaborn as sns
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
import time
import metpy.calc as mpcalc
import metpy
from metpy.units import units
from viz import dataframe_calculators as dfc
from scipy.interpolate import LinearNDInterpolator as lNDI
R = 6373.0


def error_calc(df, name, category, rmse):
    error_uj = (df['umeanh'] - df['u_scaled_approx'])
    error_vj = (df['vmeanh'] - df['v_scaled_approx'])
    speed_errorj = (error_uj**2+error_vj**2)*df['cos_weight']
    speed_errorj_sqrt = np.sqrt(error_uj**2+error_vj**2)*df['cos_weight']
    speed_errorj_sqrt_nw = np.sqrt(error_uj**2+error_vj**2)
    rmsvd = np.sqrt(speed_errorj.sum()/df['cos_weight'].sum())
    category.append(name)
    rmse.append(rmsvd)
    return speed_errorj_sqrt_nw, speed_errorj_sqrt


def random_error_add(sigma_u, sigma_v, column_u, column_v):
    e_u = np.random.normal(scale=sigma_u)
    e_v = np.random.normal(scale=sigma_v)
    e_u = np.sign(e_u)*np.minimum(2*sigma_u, abs(e_u))
    e_v = np.sign(e_v)*np.minimum(2*sigma_v, abs(e_v))

    column_u = column_u + e_u
    column_v = column_v + e_v

    return column_u, column_v


def ml_fitter(name, df,  alg, rmse, tsize, only_land, lowlat, uplat, exp_filter):

    X_train0, X_test0, y_train0, y_test0 = train_test_split(df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx',  'land', 'umeanh', 'vmeanh',  'u_error_rean', 'v_error_rean']], df[[
        'umeanh', 'vmeanh', 'land', 'lat']], test_size=tsize, random_state=1)

    sigma_u = abs(X_train0['u_error_rean'])
    sigma_v = abs(X_train0['v_error_rean'])

    X_train0['umeanh'], X_train0['vmeanh'] = random_error_add(
        sigma_u, sigma_v, X_train0['umeanh'], X_train0['vmeanh'])

    sigma_lon = 1.5
    sigma_lat = 0.15
    X_train0['lon'], X_train0['lat'] = random_error_add(
        sigma_lon, sigma_lat, X_train0['lon'], X_train0['lat'])

    print('final shape')
    print(X_train0.shape[0])
    y_train0 = X_train0[['umeanh', 'vmeanh']]
    X_train = X_train0[['lat', 'lon',
                        'u_scaled_approx', 'v_scaled_approx', 'land']]
    # X_train = X_train0[['lat', 'lon', 'land']]
    y_train = y_train0[['umeanh', 'vmeanh']]

    regressor = RandomForestRegressor(
        n_estimators=100, random_state=0, n_jobs=-1)

    print('fitting')
    start_time = time.time()
    regressor.fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    return regressor, X_test0, y_test0


def ml_predictor(name, alg, category,   rmse, tsize, lowlat, uplat, regressor, X_test0, y_test0):
        # change df0z to df for current timestep

    X_test0['cos_weight'] = np.cos(X_test0['lat']/180*np.pi)
    X_test0 = X_test0.dropna()
    y_test0 = y_test0.dropna()

    X_test0 = X_test0[(X_test0.lat >= lowlat) & (X_test0.lat <= uplat)]
    y_test0 = y_test0[(y_test0.lat >= lowlat) & (y_test0.lat <= uplat)]

    X_test = X_test0[['lat', 'lon',
                      'u_scaled_approx', 'v_scaled_approx', 'land']]
    # X_test = X_test0[['lat', 'lon', 'land']]
    y_pred = regressor.predict(X_test)

    error_u = (y_test0['umeanh'] - y_pred[:, 0])
    error_v = (y_test0['vmeanh'] - y_pred[:, 1])

    speed_error = (error_u**2+error_v**2)*X_test0['cos_weight']
    speed_error_sqrt = np.sqrt(error_u**2+error_v**2)*X_test0['cos_weight']
    speed_error_sqrt_nw = np.sqrt(error_u**2+error_v**2)

    rmsvd = np.sqrt(speed_error.sum()/X_test0['cos_weight'].sum())
    category.append(alg)
    rmse.append(rmsvd)
    X_test0['vector_diff'] = speed_error_sqrt
    X_test0['vector_diff_no_weight'] = speed_error_sqrt_nw
    if lowlat == -90 and uplat == 90:
        X_test0.to_pickle("df_rf.pkl")
    return X_test0


def error_interpolator(dfm, category, rmse):
    dfm_gt = dfm.copy()
    dfm_gt = resample(dfm_gt, replace=False,
                      n_samples=int(1e5), random_state=1)
    dfm_gt = dfm_gt[['lat', 'lon', 'u_scaled_approx',
                     'v_scaled_approx', 'umeanh', 'vmeanh', 'cos_weight', 'u_error_rean', 'v_error_rean']]
    dfm_gtf = dfm_gt.copy()
    sigma_u = abs(dfm_gt['u_error_rean'])
    sigma_v = abs(dfm_gt['v_error_rean'])

    dfm_gt['u_scaled_approx'], dfm_gt['v_scaled_approx'] = random_error_add(
        sigma_u, sigma_v, dfm_gt['umeanh'], dfm_gt['vmeanh'])

    sigma_lon = 1.5
    sigma_lat = 0.15

    dfm_gt['lon'], dfm_gt['lat'] = random_error_add(
        sigma_lon, sigma_lat, dfm_gt['lon'], dfm_gt['lat'])

    func_interp = lNDI(
        points=dfm_gt[['lat', 'lon']].values, values=dfm_gt.u_scaled_approx.values)
    dfm_gtf['u_scaled_approx'] = func_interp(
        dfm_gtf[['lat', 'lon']].values)

    func_interp = lNDI(
        points=dfm_gt[['lat', 'lon']].values, values=dfm_gt.v_scaled_approx.values)
    dfm_gtf['v_scaled_approx'] = func_interp(
        dfm_gtf[['lat', 'lon']].values)

    errors_nw, errors = error_calc(dfm_gtf, "ground_t", category, rmse)
    func_interp = lNDI(
        points=dfm_gtf[['lat', 'lon']].values, values=errors.values)
    func_interp_nw = lNDI(
        points=dfm_gtf[['lat', 'lon']].values, values=errors_nw.values)

    return dfm_gtf, func_interp_nw, func_interp


def error_rean(dfm, category, rmse):
    sigma_u = abs(dfm['u_error_rean'])
    sigma_v = abs(dfm['v_error_rean'])

    dfm['u_scaled_approx'], dfm['v_scaled_approx'] = random_error_add(
        sigma_u, sigma_v, dfm['umeanh'], dfm['vmeanh'])

    _, _ = error_calc(dfm, "ground_t", category, rmse)
    return dfm


def plot_average(deltax, df, xlist, varx, vary):
    df_mean = pd.DataFrame()
    df_unit = pd.DataFrame(data=[0], columns=[varx])
    print("calculating averages ...")
    for x in tqdm(xlist):
        df_a = df[df[varx] >= x]
        df_a = df_a[df_a[varx] <= x+deltax]
        df_unit[varx] = x
        df_a['weighted_'+vary] = df_a[vary]*df_a['cos_weight']
        df_unit[vary+'_count'] = df_a[vary].shape[0]
        df_unit[vary] = df_a['weighted_'+vary].sum()/df_a['cos_weight'].sum()
        df_a['variance'] = (df_a[vary]-df_unit[vary][0]) ** 2
        df_a['variance'] = df_a['variance']*df_a['cos_weight']
        df_unit[vary + '_std'] = np.sqrt(df_a['variance'].sum() /
                                         df_a['cos_weight'].sum())

        if df_mean.empty:
            df_mean = df_unit
        else:
            df_mean = pd.concat([df_mean, df_unit])
    return df_mean


def ds_to_netcdf(df, triplet_time, exp_filter):
    df = df.set_index(['lat', 'lon'])
    ds = df.to_xarray()
    ds = ds.rename({'u_scaled_approx': 'utrack', 'v_scaled_approx': 'vtrack'})
    ds = ds.expand_dims('time')
    ds = ds.assign_coords(time=[triplet_time])
    ds.to_netcdf('../data/processed/experiments/' +
                 exp_filter+'_'+triplet_time.strftime("%Y-%m-%d-%H:%M")+'.nc')


def latitude_selector(df, lowlat, uplat,  category, rmse, latlon, test_size, test_sizes, only_land, exp_filter, exp_list, regressor, X_test0, y_test0, triplet_time):
    dfm = df[(df.lat) <= uplat]
    dfm = df[(df.lat) >= lowlat]
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
    if exp_filter is 'exp2':
        X_test0 = ml_predictor('uv',  'rf', category, rmse, test_size, lowlat0,
                               uplat0, regressor, X_test0, y_test0)
        ds_to_netcdf(X_test0, triplet_time, exp_filter)
    elif exp_filter is 'ground_t':
        X_test0, _, _ = error_interpolator(dfm, category, rmse)
        ds_to_netcdf(X_test0, triplet_time, exp_filter)

    else:
        dfm = dfm.dropna()
        dfm['vector_diff_no_weight'], _ = error_calc(dfm, "df", category, rmse)
        dfm.to_pickle("df_df.pkl")
