import pdb
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
from data import extra_data_plotter as edp
import time
import metpy.calc as mpcalc
import metpy
from metpy.units import units
from scipy.interpolate import LinearNDInterpolator as lNDI
from scipy.interpolate import NearestNDInterpolator as NNDI
SIGMA_LON = 1.5
SIGMA_LAT = 0.15


def error_calc(df, name, category, rmse):
    """Calculates and stores error of tracker algorithm into dataframe."""

    error_uj = (df['umean'] - df['utrack'])
    error_vj = (df['vmean'] - df['vtrack'])
    speed_errorj = (error_uj**2+error_vj**2)*df['cos_weight']
    speed_errorj_sqrt = np.sqrt(error_uj**2+error_vj**2)*df['cos_weight']
    speed_errorj_sqrt_nw = np.sqrt(error_uj**2+error_vj**2)
    rmsvd = np.sqrt(speed_errorj.sum()/df['cos_weight'].sum())
    category.append(name)
    rmse.append(rmsvd)
    df['vector_diff'] = speed_errorj_sqrt
    df['vector_diff_no_weight'] = speed_errorj_sqrt_nw

    return speed_errorj_sqrt_nw, speed_errorj_sqrt, df


def random_error_add(sigma_u, sigma_v, column_u, column_v):
    """Adds random noise to dataframe.."""
    np.random.seed(1)
    e_u = np.random.normal(scale=sigma_u)
    e_v = np.random.normal(scale=sigma_v)
    e_u = np.sign(e_u)*np.minimum(2*sigma_u, abs(e_u))
    e_v = np.sign(e_v)*np.minimum(2*sigma_v, abs(e_v))

    column_u = column_u + e_u
    column_v = column_v + e_v

    return column_u, column_v


def ml_fitter(df, tsize, sigmas, with_fsua):
    """fits random forest to tracked values calculated by optical flow."""

    X_full = df[['lat', 'lon', 'utrack', 'vtrack',
                 'land', 'umean', 'vmean',  'u_error_rean', 'v_error_rean']]
    y_full = df[['umean', 'vmean', 'land', 'lat']]

    X_train0, X_test0, y_train0, y_test0 = train_test_split(
        X_full, y_full, test_size=tsize, random_state=1)

    sigma_u = abs(X_train0['u_error_rean'])
    sigma_v = abs(X_train0['v_error_rean'])

    X_train0['umean'], X_train0['vmean'] = random_error_add(
        sigma_u, sigma_v, X_train0['umean'], X_train0['vmean'])

    sigma_lon = sigmas[0]
    sigma_lat = sigmas[1]
    X_train0['lon'], X_train0['lat'] = random_error_add(
        sigma_lon, sigma_lat, X_train0['lon'], X_train0['lat'])

    print('final shape')
    print(X_train0.shape[0])
    y_train = X_train0[['umean', 'vmean']]
    if with_fsua:
        X_train = X_train0[['lat', 'lon',
                            'utrack', 'vtrack', 'land']]
    else:
        X_train = X_train0[['lat', 'lon', 'land']]

    regressor = RandomForestRegressor(
        n_estimators=100, random_state=0, n_jobs=-1)

    print('fitting')
    start_time = time.time()
    regressor.fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    return regressor, X_test0, y_test0, X_full


def ml_predictor(category, name, rmse,  regressor, X_test0, y_test0, X_full0, with_fsua):
    """Corrects optical flow velocities with random forest model."""

    X_test0['cos_weight'] = np.cos(X_test0['lat']/180*np.pi)
    X_full0['cos_weight'] = np.cos(X_full0['lat']/180*np.pi)

    X_test0 = X_test0.dropna()
    y_test0 = y_test0.dropna()
    X_full0 = X_full0.dropna()

    if with_fsua:
        X_test = X_test0[['lat', 'lon',
                          'utrack', 'vtrack', 'land']]
        X_full = X_full0[['lat', 'lon',
                          'utrack', 'vtrack', 'land']]
    else:
        X_test = X_test0[['lat', 'lon', 'land']]
        X_full = X_full0[['lat', 'lon', 'land']]

    y_pred = regressor.predict(X_test)
    y_pred_full = regressor.predict(X_full)

    X_test0['utrack'] = y_pred[:, 0]
    X_test0['vtrack'] = y_pred[:, 1]

    X_full0['utrack'] = y_pred_full[:, 0]
    X_full0['vtrack'] = y_pred_full[:, 1]

    _, _, X_test0 = error_calc(X_test0, name, category, rmse)
    _, _, X_full0 = error_calc(X_full0, name, [], [])

    return X_test0, X_full0


def error_interpolator(dfm, category, name, rmse, sigmas):
    """Interpolates error into appropriate coordinates, since error is calculated by adding noise to coordinates."""

    dfm_gt = dfm.copy()
   # dfm_gt = resample(dfm_gt, replace=False,
    #                  n_samples=int(1e5), random_state=1)
    dfm_gt = dfm_gt[['lat', 'lon', 'utrack',
                     'vtrack', 'umean', 'vmean', 'cos_weight', 'u_error_rean', 'v_error_rean']]
    dfm_gtf = dfm_gt.copy()
    sigma_u = abs(dfm_gt['u_error_rean'])
    sigma_v = abs(dfm_gt['v_error_rean'])

    dfm_gt['utrack'], dfm_gt['vtrack'] = random_error_add(
        sigma_u, sigma_v, dfm_gt['umean'], dfm_gt['vmean'])

    sigma_lon = sigmas[0]
    sigma_lat = sigmas[1]

    dfm_gt['lon'], dfm_gt['lat'] = random_error_add(
        sigma_lon, sigma_lat, dfm_gt['lon'], dfm_gt['lat'])

    func_interp = NNDI(
        x=dfm_gt[['lat', 'lon']].values, y=dfm_gt.utrack.values)
    dfm_gtf['utrack'] = func_interp(
        dfm_gtf[['lat', 'lon']].values)

    func_interp = NNDI(
        x=dfm_gt[['lat', 'lon']].values, y=dfm_gt.vtrack.values)
    dfm_gtf['vtrack'] = func_interp(
        dfm_gtf[['lat', 'lon']].values)

    _, _, dfm_gtf = error_calc(dfm_gtf, name, category, rmse)

    return dfm_gtf


def error_rean(dfm, category, rmse):
    """Adds error from reanalysis to GEOS-5 ground truth."""

    sigma_u = abs(dfm['u_error_rean'])
    sigma_v = abs(dfm['v_error_rean'])

    dfm['utrack'], dfm['vtrack'] = random_error_add(
        sigma_u, sigma_v, dfm['umean'], dfm['vmean'])

    _, _ = error_calc(dfm, "ground_t", category, rmse)
    return dfm


def ds_to_netcdf(df, triplet_time, exp_filter):
    """Saves dataset to netCDF file."""

    df = df.set_index(['lat', 'lon'])
    ds = df.to_xarray()
    ds = ds[['umean', 'vmean', 'utrack',
             'vtrack', 'cos_weight', 'u_error_rean', 'v_error_rean']]
    ds = ds.rename({'utrack': 'utrack', 'vtrack': 'vtrack'})
    ds = ds.expand_dims('time')
    ds = ds.assign_coords(time=[triplet_time])
    ds.to_netcdf('../data/processed/experiments/' + exp_filter +
                 '_'+triplet_time.strftime("%Y-%m-%d-%H:%M")+'.nc')


def random_forest_calculator(df,  category, rmse,   exp_filter, exp_list, regressor, X_test0, y_test0, triplet_time, X_full, sigmas, with_fsua):
    """Calculates second stage of UA algorithm."""
    exp_list.append(exp_filter)

    if exp_filter is 'exp2':
        X_test0, X_full = ml_predictor(category, 'rf',
                                       rmse,  regressor, X_test0, y_test0, X_full, with_fsua)
        ds_to_netcdf(X_test0, triplet_time, exp_filter)
        ds_to_netcdf(X_full, triplet_time, 'full_' + exp_filter)

    elif exp_filter is 'ground_t':
        X_test0 = error_interpolator(df, category, exp_filter, rmse, sigmas)
        ds_to_netcdf(X_test0, triplet_time, exp_filter)

    else:
        _, _, df = error_calc(df, exp_filter, category, rmse)
        ds_to_netcdf(df, triplet_time, exp_filter)
