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
import transport_calculators as tc
import cv2
from pathlib import Path

COORDS = [(-30, 30), (-60, -30), (-90, -60), (30, 60), (60, 90)]
KERNEL = 5
# KERNEL=20


def grad_calculator(ds, dy, dx, date, kernel):
    u = ds['umean'].sel(time=str(date)).values
    v = ds['vmean'].sel(time=str(date)).values
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
    ds_unit['utrack'] = ds['utrack']
    ds_unit['vtrack'] = ds['vtrack']

    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'vorticity': da4})])

    u = ds['utrack'].sel(time=str(date)).values
    v = ds['vtrack'].sel(time=str(date)).values
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
file_name = '3600_850_july.nc'
files = []
file1 = '../data/processed/experiments/'+file_name
#file2 = '../data/interim/experiments/july/tracked/60min/combined/850_july.nc'
files.append(file1)


for file in files:
    ds = xr.open_dataset(file)
    print(ds)
    lat = ds.lat.values
    lon = ds.lon.values
    print('calculating deltas...')
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

    ds_tot = xr.Dataset()
    times = ds.time.values
    times = [times[0]]
    kernels = [0, 5, 10, 20]
    kernels = [44]
    for kernel in kernels:
        ds_tot = xr.Dataset()
        rmses = []
        region = []
        filter_res = []

        for date in times:
            print(date)
            ds_unit = grad_calculator(ds, dy, dx, date, kernel)
            if len(ds_tot) > 0:
                ds_tot = xr.concat([ds_tot, ds_unit], 'time')
            else:
                ds_tot = ds_unit
        df = ds_tot.to_dataframe().dropna().reset_index()
        print('shape ' + str(df.shape))
        tc.rmse_lists(df, rmses, region, filter_res)
        d = {'latlon': region, 'exp_filter': filter_res, 'rmse': rmses}
        df_results = pd.DataFrame(data=d)
        kernel = str(kernel)
        name = kernel+'_'+Path(file).stem

        print("kernel: " + kernel)
        print(df_results)
        tc.filter_plotter(df_results, 'div_vort_'+name, ' ')
        ds_tot = ds_tot.sel(time=str(ds_tot.time.values[0]))
        # ds_tot['divergence'] = abs(ds_tot.divergence)
        # ds_tot['vorticity'] = abs(ds_tot.vorticity)
        tc.plotter(ds_tot, 'error_div', '850', '850', name)
        tc.plotter(ds_tot, 'error_vort', '850', '850', name)
        tc.plotter(ds_tot, 'divergence', '850', '850', name)
        tc.plotter(ds_tot, 'vorticity', '850', '850', name)
