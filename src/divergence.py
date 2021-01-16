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
import cv2


def quiver_plotter(ds, title, ulabel, vlabel):
    ds = ds.coarsen(lon=5, boundary='trim').mean().coarsen(
        lat=5, boundary='trim').mean()
    fig, ax = plt.subplots()
    fig.tight_layout()
    Q = ax.quiver(np.squeeze(ds['lon'].values), np.squeeze(
        ds['lat'].values), np.squeeze(ds[ulabel].values), np.squeeze(ds[vlabel].values))

    ax.set_title('Observed Velocities')
    plt.savefig(title+'.png', bbox_inches='tight', dpi=300)
    print('plotted quiver...')


def grad_quants(ds, ulabel, vlabel, dx, dy, kernel):
    u = ds[ulabel].values
    v = ds[vlabel].values
    u = np.squeeze(u)
    v = np.squeeze(v)
    u, v, div = tc.div_calc(
        u.copy(), v.copy(), dx.copy(), dy.copy(), kernel, True)
    u, v, vort = tc.vort_calc(
        u.copy(), v.copy(), dx.copy(), dy.copy(), kernel, True)
    return div, vort, u, v


def grad_calculator(ds,  kernel):
    lat = ds.lat.values
    lon = ds.lon.values
    print('calculating deltas...')

    u, v = tc.vel_filter(np.squeeze(
        ds['umean'].values), np.squeeze(ds['vmean'].values))
    ds['umean_s'] = (['lat', 'lon'], u)
    ds['vmean_s'] = (['lat', 'lon'], v)
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    div, vort, u, v = grad_quants(ds, 'umean_s', 'vmean_s', dx, dy, kernel)
    ds['divergence'] = (['lat', 'lon'], div)
    ds['vorticity'] = (['lat', 'lon'], vort)

    u, v = tc.vel_filter(np.squeeze(
        ds['utrack'].values), np.squeeze(ds['vtrack'].values))
    ds['utrack_s'] = (['lat', 'lon'], u)
    ds['vtrack_s'] = (['lat', 'lon'], v)
    div, vort, u, v = grad_quants(ds, 'utrack_s', 'vtrack_s', dx, dy, kernel)
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
    kernel = 3
    ds = xr.open_dataset(file)
    ds = ds.loc[{'lat': slice(5, 15), 'lon': slice(-138, -112)}]
   # ds = ds.coarsen(lat=10, boundary='trim').mean().coarsen(
    #    lon=10, boundary='trim').mean()
    ds = grad_calculator(ds, kernel)
    print(ds)
    print(abs(ds['vorticity']).mean())
    print(abs(ds['divergence']).mean())

    quiver_plotter(ds, 'groundt_s', 'umean_s', 'vmean_s')
    quiver_plotter(ds, 'track_s', 'utrack_s', 'vtrack_s')
    quiver_plotter(ds, 'groundt', 'umean', 'vmean')
    quiver_plotter(ds, 'track', 'utrack', 'vtrack')


if __name__ == "__main__":
    main()
