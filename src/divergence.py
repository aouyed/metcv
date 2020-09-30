from pathlib import Path
import cv2
import transport_calculators as tc
from data import extra_data_plotter as edp
from data import map_maker as mm
import matplotlib.pyplot as plt
import xarray as xr
import pdb
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime
import numpy as np
import pandas as pd

COORDS = [(-30, 30), (-60, -30), (-90, -60), (30, 60), (60, 90)]
KERNEL = 5
# KERNEL=20


def artificial_track(ds_unit, factor):
    sigma_u = factor*ds_unit['error_u'].values
    sigma_v = factor*ds_unit['error_v'].values
    e_u = np.random.normal(scale=sigma_u)
    e_v = np.random.normal(scale=sigma_v)
    e_u = np.sign(e_u)*np.minimum(2*sigma_u, abs(e_u))
    e_v = np.sign(e_v)*np.minimum(2*sigma_v, abs(e_v))
    u = ds_unit['umean'].values
    v = ds_unit['vmean'].values
    utracka = u + e_u
    vtracka = v + e_v
    ds_unit['utrack_a'] = (['lat', 'lon'], np.squeeze(utracka))
    ds_unit['vtrack_a'] = (['lat', 'lon'], np.squeeze(vtracka))
    return ds_unit


def grad_calculator(ds, dy, dx,  kernel, utrack_name, vtrack_name):
    u = ds['umean'].values
    v = ds['vmean'].values
    u = np.squeeze(u)
    v = np.squeeze(v)
    div = tc.div_calc(u.copy(), v.copy(), dx.copy(), dy.copy(), kernel, False)
    vort = tc.vort_calc(u, v, dx, dy, kernel, False)
    print('building data arrays...')
    da3 = tc.build_datarray(div, lat, lon, date)
    da4 = tc.build_datarray(vort, lat, lon, date)
    ds_unit = xr.Dataset({'divergence': da3})
    ds_unit['umean'] = ds['umean']
    ds_unit['vmean'] = ds['vmean']
    ds_unit[utrack_name] = ds[utrack_name]
    ds_unit[vtrack_name] = ds[vtrack_name]

    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'vorticity': da4})])

    u = ds[utrack_name].values
    v = ds[vtrack_name].values
    u = np.squeeze(u)
    v = np.squeeze(v)

    div = tc.div_calc(u.copy(), v.copy(), dx.copy(), dy.copy(), kernel, True)
    vort = tc.vort_calc(u, v, dx, dy, kernel, True)
    da3 = tc.build_datarray(div, lat, lon, date)
    da4 = tc.build_datarray(vort, lat, lon, date)
    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'divergence_track': da3})])
    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'vorticity_track': da4})])
    ds_unit['cos_weight'] = np.cos(ds_unit.lat/180*np.pi)
    # ds_unit['cos_weight'] = ds['cos_weight'].sel(time=str(date))
    ds_unit['error_div'] = abs(ds_unit.divergence-ds_unit.divergence_track)
    ds_unit['error_vort'] = abs(ds_unit.vorticity-ds_unit.vorticity_track)
    print(ds_unit)
    return ds_unit


months = [7]
pressures = [850, 500]
dts = [3600]
file_name = '3600_850_full_july.nc'
files = []
file1 = '../data/processed/experiments/'+file_name
#file2 = '../data/interim/experiments/july/tracked/60min/combined/850_july.nc'
files.append(file1)


for file in files:
    ds = xr.open_dataset(file)
    lat = ds.lat.values
    lon = ds.lon.values
    #ds = ds.sel(filter='exp2')
    print('calculating deltas...')
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

    ds_tot = xr.Dataset()
    times = ds.time.values
    times = [times[0]]
    kernels = [0, 5, 10, 20]
    kernel = 0
    factors = [0.06, 0.12, 0.25, 0.5, 1, 2, 10]
    #factors = [10]
    utrack_name = 'utrack_a'
    vtrack_name = 'vtrack_a'
    for factor in factors:
        print('factor: ' + str(factor))
        ds_tot = xr.Dataset()
        rmses = []
        region = []
        filter_res = []

        for date in times:
            print(date)
            ds_unit = ds.sel(time=str(date))
            ds_unit['error_u'] = abs(ds_unit['umean']-ds_unit['utrack'])
            ds_unit['error_v'] = abs(ds_unit['vmean']-ds_unit['vtrack'])
            ds_unit = artificial_track(ds_unit, factor)

            ds_unit = grad_calculator(
                ds_unit, dy, dx,  kernel,  utrack_name, vtrack_name)
            if len(ds_tot) > 0:
                ds_tot = xr.concat([ds_tot, ds_unit], 'time')
            else:
                ds_tot = ds_unit
        df = ds_tot.to_dataframe().dropna().reset_index()
        print('shape ' + str(df.shape))
        tc.rmse_lists(df, rmses, region, filter_res, utrack_name, vtrack_name)
        d = {'latlon': region, 'exp_filter': filter_res, 'rmse': rmses}
        df_results = pd.DataFrame(data=d)
        name = str(factor)+'_'+str(kernel)+'_'+Path(file).stem

        print(df_results)
        df_results = edp.sorting_latlon(df_results)

        tc.filter_plotter(df_results, 'div_vort_'+name, ' ')
        ds_tot = ds_tot.sel(time=str(ds_tot.time.values[0]))
        # ds_tot['divergence'] = abs(ds_tot.divergence)
        # ds_tot['vorticity'] = abs(ds_tot.vorticity)
        tc.plotter(ds_tot, 'error_div', '850', '850', name)
        tc.plotter(ds_tot, 'error_vort', '850', '850', name)
        tc.plotter(ds_tot, 'divergence', '850', '850', name)
        tc.plotter(ds_tot, 'vorticity', '850', '850', name)
        tc.plotter(ds_tot, 'vorticity_track', '850', '850', name)
