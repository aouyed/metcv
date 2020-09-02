import xarray as xr
import pdb
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime
import numpy as np


def build_datarray(data, lat, lon, date):
    da = xr.DataArray(data, coords=[
        lat, lon], dims=['lat', 'lon'])
    da = da.expand_dims('time')
    da = da.assign_coords(time=[date])
    return da


def grad_calculator(ds, dy, dx, date):
    u = ds['umean'].sel(time=str(date)).values
    v = ds['vmean'].sel(time=str(date)).values
    u = np.squeeze(u)
    v = np.squeeze(v)
    div = div_calc(u, v, dx, dy)
    vort = vort_calc(u, v, dx, dy)
    print('building data arrays...')
    da3 = build_datarray(div, lat, lon, date)
    da4 = build_datarray(vort, lat, lon, date)
    ds_unit = xr.Dataset({'divergence': da3})
    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'vorticity': da4})])

    u = ds['utrack'].sel(time=str(date)).values
    v = ds['vtrack'].sel(time=str(date)).values
    u = np.squeeze(u)
    v = np.squeeze(v)
    div = div_calc(u, v, dx, dy)
    vort = vort_calc(u, v, dx, dy)
    da3 = build_datarray(div, lat, lon, date)
    da4 = build_datarray(vort, lat, lon, date)
    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'divergence_track': da3})])
    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'vorticity_track': da4})])
    print(ds_unit)
    return ds_unit


def div_calc(u, v, dx, dy):
    div = mpcalc.divergence(
        u * units['m/s'], v * units['m/s'], dx, dy, dim_order='yx')
    div = div.magnitude
    #div = np.nan_to_num(div)
    return div


def vort_calc(u, v, dx, dy):
    vort = mpcalc.vorticity(
        u * units['m/s'], v * units['m/s'], dx, dy, dim_order='yx')
    vort = vort.magnitude
    #vort = np.nan_to_num(vort)

    return vort


months = [7]
pressures = [850, 500]
dts = [3600]
for dt in dts:
    for month in months:
        for pressure in pressures:
            for day in (1, 2, 3):
                if month == 1:
                    month_str = 'january'
                else:
                    month_str = 'july'

                print(month_str)
                file_name = '3600_850_full_july.nc'

                file = '../../data/processed/experiments/'+file_name
                print(file)
                ds = xr.open_dataset(file)
                print(ds)
                lat = ds.lat.values
                lon = ds.lon.values
                print('calculating deltas...')
                dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

                ds_tot = xr.Dataset()
                for date in ds.time.values:
                    print(date)
                    ds_unit = grad_calculator(ds, dy, dx, date)
                    if len(ds_tot) > 0:
                        ds_tot = xr.concat([ds_tot, ds_unit], 'time')
                    else:
                        ds_tot = ds_unit
                print(ds_tot)
                breakpoint()
