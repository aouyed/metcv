from pathlib import Path

import transport_calculators as tc
import matplotlib.pyplot as plt
import xarray as xr
import pdb
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime
import numpy as np
import pandas as pd


def grad_quants(ds, ulabel, vlabel, dx, dy, kernel):
    u = ds[ulabel].values
    v = ds[vlabel].values
    u = np.squeeze(u)
    v = np.squeeze(v)
    div = tc.div_calc(u.copy(), v.copy(), dx.copy(), dy.copy(), kernel, True)
    vort = tc.vort_calc(u, v, dx, dy, kernel, True)
    return div, vort


def grad_calculator(ds,  kernel):
    lat = ds.lat.values
    lon = ds.lon.values
    print('calculating deltas...')
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    div, vort = grad_quants(ds, 'umean', 'vmean', dx, dy, kernel)
    ds['divergence'] = (['lat', 'lon'], div)
    ds['vorticity'] = (['lat', 'lon'], vort)

    div, vort = grad_quants(ds, 'utrack', 'vtrack', dx, dy, kernel)
    ds['divergence_track'] = (['lat', 'lon'], div)
    ds['vorticity_track'] = (['lat', 'lon'], vort)

    ds['error_div_abs'] = abs(ds.divergence-ds.divergence_track)
    ds['error_vort_abs'] = abs(ds.vorticity-ds.vorticity_track)
    ds['error_div_sqr'] = (
        ds.divergence-ds.divergence_track)**2*ds['cos_weight']
    ds['error_vort_sqr'] = (
        ds.vorticity-ds.vorticity_track)**2*ds['cos_weight']
    ds['rmsvd_vort'] = np.sqrt(
        ds['error_vort_sqr'].sum()/ds['cos_weight'].sum())
    ds['rmsvd_div'] = np.sqrt(ds['error_div_sqr'].sum()/ds['cos_weight'].sum())

    return ds


def main():

    file = '../data/processed/experiments/900_700_full_november.nc'
    kernel = 0
    ds = xr.open_dataset(file)
   # ds = ds.coarsen(lat=4, boundary='trim').mean().coarsen(
    #    lon=4, boundary='trim').mean()
    ds = grad_calculator(ds, kernel)
    print(ds)
    breakpoint()


if __name__ == "__main__":
    main()
