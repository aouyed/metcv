import xarray as xr
import pdb
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import map_maker as mm
from data import extra_data_plotter as edp
import cv2
import cartopy.crs as ccrs
import pickle
import seaborn as sns
import cmocean
import matplotlib.colors as mcolors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

COORDS = [(-30, 30), (-60, -30), (-90, -60), (30, 60), (60, 90)]
KERNEL = 5
# KERNEL=20
SCALE = 1e4
PASSES = 200


def map_plotter(var, title, units, vmin, vmax):
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    # pmap = plt.cm.gnuplot
    pmap = cmocean.cm.thermal
    # pmap = plt.cm.coolwarm
    # pmap.set_bad(color='grey')
    if abs(vmax) > 0:
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vmax/4, vmax=vmax)
        im = ax.imshow(var, cmap=pmap,
                       extent=[-180, 180, -90, 90], origin='lower', vmin=vmin, vmax=vmax, norm=divnorm)
    else:
        im = ax.imshow(
            var, cmap=pmap, extent=[-180, 180, -90, 90], origin='lower')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    cbar.set_label(units)
    plt.xlabel("lon")
    plt.ylabel("lat")
    # ax.set_title(title)
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
    plt.close()


def plotter(ds, varname, dt, pressure, filter):
    var = ds[varname].values
    var = np.squeeze(var)
    print('test')
    print(np.mean(np.nan_to_num(var)))
    vmin = -0.25
    vmax = 1
    # if varname == 'error_vort':
    #   vmin = 0
    #  vmax = 0.006
    # if varname == vorticity':
    #   var = np.maximum(var, 6)
    #  var = np.minimum(var, -6)
    map_plotter(var, dt+'_'+str(pressure) + '_' +
                varname + '_' + filter, 'm/s', vmin, vmax)


def filter_plotter(df0, values, title):
    fig, ax = plt.subplots()
    filters = ['vorticity_rmse', 'abs_vorticity_mean']

    for filter in filters:
        df = df0[df0.exp_filter == filter]
        ax.plot(df['latlon'], df['rmse'], '-o', label=filter)

    ax.legend(frameon=None)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSE [1/s]")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)
    plt.close()


def build_datarray(data, lat, lon, date):
    da = xr.DataArray(data, coords=[
        lat, lon], dims=['lat', 'lon'])
    da = da.expand_dims('time')
    da = da.assign_coords(time=[date])
    return da


def div_calc(u, v, dx, dy, kernel, is_track):
    div = mpcalc.divergence(
        u * units['m/s'], v * units['m/s'], dx, dy, dim_order='yx')
    div = div.magnitude
    div = SCALE*div
    return u, v, div


def vort_calc(u, v, dx, dy, kernel, is_track):
    vort = mpcalc.vorticity(
        u * units['m/s'], v * units['m/s'], dx, dy, dim_order='yx')
    vort = vort.magnitude
    vort = SCALE*vort
    #vort = cv2.blur(np.nan_to_num(vort), (kernel, kernel))

    return u, v, vort


def vel_filter(u, v):
    start_time = time.time()

    u = np.float32(u)
    v = np.float32(v)
    mask_u = np.isnan(u)
    mask_v = np.isnan(v)

    u = np.nan_to_num(u)
    v = np.nan_to_num(v)

    v = cv2.medianBlur(v, 5)
    u = cv2.medianBlur(u, 5)

    u = mpcalc.smooth_n_point(u, 9, PASSES)
    v = mpcalc.smooth_n_point(v, 9, PASSES)

    u[mask_u] = np.nan
    v[mask_v] = np.nan
    print("--- seconds ---" + str(time.time() - start_time))

    return u, v


def vort_filter(u, v):
    start_time = time.time()

    u = np.float32(u)
    v = np.float32(v)
    mask_u = np.isnan(u)
    mask_v = np.isnan(v)

    u = np.nan_to_num(u)
    v = np.nan_to_num(v)

    #v = cv2.medianBlur(v, 5)
    #u = cv2.medianBlur(u, 5)

    u = mpcalc.smooth_n_point(u, 9, PASSES)
    v = mpcalc.smooth_n_point(v, 9, PASSES)

    u[mask_u] = np.nan
    v[mask_v] = np.nan
    print("--- seconds ---" + str(time.time() - start_time))

    return u, v


def coord_to_string(coord):
    if coord[0] < 0:
        lowlat = str(abs(coord[0])) + '°S'
    else:
        lowlat = str(coord[0]) + '°N'

    if coord[1] < 0:
        uplat = str(abs(coord[1])) + '°S'
    else:
        uplat = str(coord[1]) + '°N'
    stringd = str(str(lowlat)+',' + str(uplat))
    return stringd


def rmse_lists(df, rmses, region, filter_res, utrack_name, vtrack_name):
    for coord in COORDS:

        rmse_div, rmse_vort, rmsvd,  mean_div, mean_vort = rmse_calculator(
            df, coord, utrack_name, vtrack_name)

        stringc = coord_to_string(coord)
        filter_res.append('rmsvd')
        rmses.append(rmsvd)
        region.append(stringc)
        filter_res.append('divergence_rmse')
        rmses.append(rmse_div)
        region.append(stringc)
        filter_res.append('vorticity_rmse')
        rmses.append(rmse_vort)
        region.append(stringc)
        filter_res.append('abs_divergence_mean')
        rmses.append(mean_div)
        region.append(stringc)
        filter_res.append('abs_vorticity_mean')
        rmses.append(mean_vort)
        region.append(stringc)


def error_calc(df, utrack_name, vtrack_name):
    """Calculates and stores error of tracker algorithm into dataframe."""

    error_uj = (df['umean'] - df[utrack_name])
    error_vj = (df['vmean'] - df[vtrack_name])
    speed_errorj = (error_uj**2+error_vj**2)*df['cos_weight']
    rmsvd = np.sqrt(speed_errorj.sum()/df['cos_weight'].sum())
    return rmsvd


def rmse_calculator(df, coord, utrack_name, vtrack_name):
    df_unit = df[(df.lat >= coord[0]) & (df.lat <= coord[1])]
    df_unit['delta_div'] = df_unit.cos_weight * \
        (df_unit.divergence-df_unit.divergence_track)**2
    df_unit['delta_vort'] = df_unit.cos_weight * \
        (df_unit.vorticity-df_unit.vorticity_track)**2
    df_unit['weighed_abs_div'] = df_unit.cos_weight*abs(df_unit.divergence)
    df_unit['weighed_abs_vort'] = df_unit.cos_weight*abs(df_unit.vorticity)

    rmse_div = np.sqrt(df_unit['delta_div'].sum()/df_unit['cos_weight'].sum())
    rmse_vort = np.sqrt(df_unit['delta_vort'].sum() /
                        df_unit['cos_weight'].sum())
    rmsvd = error_calc(df_unit.copy(), utrack_name, vtrack_name)
    mean_div = df_unit['weighed_abs_div'].sum()/df_unit['cos_weight'].sum()
    mean_vort = df_unit['weighed_abs_vort'].sum()/df_unit['cos_weight'].sum()

    return rmse_div, rmse_vort, rmsvd, mean_div, mean_vort


def ml_fitter(ds, tsize):
    """fits random forest to tracked values calculated by optical flow."""
    df = ds.to_dataframe()
    df = df.reset_index()
    df = df.dropna()
    X_full = df[['lat', 'lon', 'vorticity_track', 'utrack', 'vtrack']]
    y_full = df[['vorticity']]

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=tsize, random_state=1)

    regressor = RandomForestRegressor(
        n_estimators=100, random_state=0, n_jobs=-1)

    print('fitting')
    start_time = time.time()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    y_pred_full = regressor.predict(X_full)

    X_test['vorticity_track'] = y_pred
    X_test = X_test.set_index(['lat', 'lon'])
    X_full['vorticity_track'] = y_pred_full
    X_full = X_full.set_index(['lat', 'lon'])

    ds_test = xr.Dataset.from_dataframe(X_test)
    ds_full = xr.Dataset.from_dataframe(X_full)
    ds['vorticity_track'] = ds_test['vorticity_track']
    ds['vorticity_track_full'] = ds_full['vorticity_track']
    print("--- %s seconds ---" % (time.time() - start_time))

    return ds
