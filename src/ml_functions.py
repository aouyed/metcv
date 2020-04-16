
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


def vorticity(df):
    print('Calculating vorticity...')
    u_a = df.pivot('lat', 'lon', 'umeanh').values
    v_a = df.pivot('lat', 'lon', 'vmeanh').values
    # lon = df.pivot('lat', 'lon', 'lon').values
    # lat = df.pivot('lat', 'lon', 'lat').values
    u_a = np.nan_to_num(u_a)
    v_a = np.nan_to_num(v_a)
    # print(np.)
    # print(np.max(lon))
    lon = np.arange(df['lon'].min(), df['lon'].max() + 0.0625, 0.0625)
    lat = np.arange(df['lat'].min(), df['lat'].max() + 0.0625, 0.0625)

    dx, dy = metpy.calc.lat_lon_grid_deltas(lon, lat)
    omega = mpcalc.vorticity(u_a * units['m/s'],
                             v_a * units['m/s'], dx, dy, dim_order='yx')

    omega = omega.magnitude

    df_u = pd.DataFrame(omega).stack().rename_axis(
        ['y', 'x']).reset_index(name='vorticity')
    df_u = dfc.latlon_converter(df_u, 0.0625)
    df_u['vorticity'] = df_u['vorticity']/(1e-5)
    df = df.merge(df_u, how='left')
    return df


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


def df_freq(df, values, title):
    print('calculating frequency...')
    # freq_group = df[values]
    freq_group = df.groupby(values).size()
    freq_group = freq_group.reset_index()
    freq_group = freq_group.rename(columns={0: 'freq'})
    print(freq_group)
  #  freq_group['freq'] = freq_group['freq']
    # print("plotting...")
    edp.freq_plotter(freq_group, values, title)


def random_error_add(sigma_u, sigma_v, column_u, column_v):
    e_u = np.random.normal(scale=sigma_u)
    e_v = np.random.normal(scale=sigma_v)
    e_u = np.sign(e_u)*np.minimum(2*sigma_u, abs(e_u))
    e_v = np.sign(e_v)*np.minimum(2*sigma_v, abs(e_v))

    column_u = column_u + e_u
    column_v = column_v + e_v

    return column_u, column_v


def ml_fitter(name, f, df,  alg, rmse, tsize, only_land, lowlat, uplat, exp_filter):

    X_train0, X_test0, y_train0, y_test0 = train_test_split(df[['lat', 'lon', 'u_scaled_approx', 'v_scaled_approx', 'utrack', 'land', 'sample_weight', 'umeanh', 'vmeanh', 'distance']], df[[
        'umeanh', 'vmeanh', 'utrack', 'land', 'lat']], test_size=tsize, random_state=1)

    df_freq(X_train0, 'distance', 'nosample')
    deltax = 100
    maxr = np.pi*R
    exp_distance = np.exp(2*(X_train0.distance)/(np.pi*R))

    sigma_u = abs(2*exp_distance)
    sigma_v = abs(0.2*exp_distance)

    X_train0['umeanh'], X_train0['vmeanh'] = random_error_add(
        sigma_u, sigma_v, X_train0['umeanh'], X_train0['vmeanh'])

    sigma_lon = 2*0.625*exp_distance
    sigma_lat = 2*0.0625*exp_distance

    X_train0['lon'], X_train0['lat'] = random_error_add(
        sigma_lon, sigma_lat, X_train0['lon'], X_train0['lat'])

    print('final shape')
    print(X_train0.shape[0])
    df_freq(X_train0, 'distance', 'rsample')
    y_train0 = X_train0[['umeanh', 'vmeanh']]
    sample_weight = X_train0['sample_weight']
    X_train = X_train0[['lat', 'lon',
                        'u_scaled_approx', 'v_scaled_approx', 'land']]
    #X_train = X_train0[['lat', 'lon']]
    y_train = y_train0[['umeanh', 'vmeanh']]

    regressor = RandomForestRegressor(
        n_estimators=100, random_state=0, n_jobs=-1)

    print('fitting')
    start_time = time.time()
    regressor.fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
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
    #X_test = X_test0[['lat', 'lon']]
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
    if exp_filter is 'exp2':
        ml_predictor('uv', f, 'rf', category, rmse, test_size, lowlat0,
                     uplat0, regressor, X_test0, y_test0)
    else:
        dfm_gt = dfm.copy()

        dfm_gt = resample(dfm_gt, replace=False,
                          n_samples=int(1e5), random_state=1)

        dfm_gt = dfm_gt[['lat', 'lon', 'u_scaled_approx',
                         'v_scaled_approx', 'umeanh', 'vmeanh', 'distance', 'cos_weight']]
        dfm_gtf = dfm_gt.copy()
        exp_distance = np.exp(2*(dfm_gt.distance)/(np.pi*R))

        sigma_u = 2*exp_distance
        sigma_v = 0.2*exp_distance

        dfm_gt['u_scaled_approx'], dfm_gt['v_scaled_approx'] = random_error_add(
            sigma_u, sigma_v, dfm_gt['umeanh'], dfm_gt['vmeanh'])

        sigma_lon = 2*0.625*exp_distance
        sigma_lat = 2*0.0625*exp_distance

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

        error_calc(dfm_gtf, f, "ground_t", category, rmse)

    dfm = dfm.dropna()
    latlon.append(str(str(lowlat)+',' + str(uplat)))
    test_sizes.append(test_size)
    exp_list.append(exp_filter)

    error_calc(dfm, f, "df", category, rmse)
    latlon.append(str(str(lowlat)+',' + str(uplat)))
    test_sizes.append(test_size)
    exp_list.append(exp_filter)

    error_calc(dftm, f, 'jpl', category, rmse)
