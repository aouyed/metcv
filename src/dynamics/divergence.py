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

PATH = '../data/processed/plots/dynamics/'


def quiver_plotter(ds, title, ulabel, vlabel):

    ds = ds.coarsen(lon=15, boundary='trim').mean().coarsen(
        lat=15, boundary='trim').mean()
    fig, ax = plt.subplots()
    fig.tight_layout()
    Q = ax.quiver(np.squeeze(ds['lon'].values), np.squeeze(
        ds['lat'].values), np.squeeze(ds[ulabel].values), np.squeeze(ds[vlabel].values))
    qk = ax.quiverkey(Q, 0.9, 0.9, 10, r'$10 \frac{m}{s}$', labelpos='E',
                      coordinates='figure')
    ax.set_title('Observed Velocities')
    plt.savefig(title+'.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('plotted quiver...')


def grad_quants(ds, ulabel, vlabel, dx, dy, passes, threshold):
    u = ds[ulabel].values
    v = ds[vlabel].values
    u = np.squeeze(u)
    v = np.squeeze(v)
    u, v, div = tc.div_calc(
        u.copy(), v.copy(), dx.copy(), dy.copy(), passes, True)
    u, v, vort = tc.vort_calc(
        u.copy(), v.copy(), dx.copy(), dy.copy(), passes, True)

    if threshold > 0 and ulabel == 'utrack_s':
        print('threshold passed')
        div[abs(div) < threshold] = np.nan
        vort[abs(vort) < threshold] = np.nan
    return div, vort, u, v


def map_plotter(ds, title, label, units_label, vmin, vmax):
    values = np.squeeze(ds[label].values)
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap='viridis', extent=[ds['lon'].min(
    ), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()], vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(units_label)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(title+'png', dpi=300)
    plt.close()


def grad_calculator(ds,  passes, threshold):
    lat = ds.lat.values
    lon = ds.lon.values
    print('calculating deltas...')

    u, v = tc.vel_filter(np.squeeze(
        ds['umean'].values), np.squeeze(ds['vmean'].values), passes)
    ds['umean_s'] = (['lat', 'lon'], u)
    ds['vmean_s'] = (['lat', 'lon'], v)
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    div, vort, u, v = grad_quants(
        ds, 'umean_s', 'vmean_s', dx, dy, passes, threshold)
    ds['divergence'] = (['lat', 'lon'], div)
    ds['vorticity'] = (['lat', 'lon'], vort)

    u, v = tc.vel_filter(np.squeeze(
        ds['utrack'].values), np.squeeze(ds['vtrack'].values), passes)
    ds['utrack_s'] = (['lat', 'lon'], u)
    ds['vtrack_s'] = (['lat', 'lon'], v)
    div, vort, u, v = grad_quants(
        ds, 'utrack_s', 'vtrack_s', dx, dy, passes, threshold)

    ds['divergence_track'] = (['lat', 'lon'], div)
    ds['vorticity_track'] = (['lat', 'lon'], vort)

    ds['u_error'] = ds['utrack']-ds['umean']
    ds['u_error_s'] = ds['utrack_s']-ds['umean_s']
    ds['v_error'] = ds['vtrack']-ds['vmean']
    ds['v_error_s'] = ds['vtrack_s']-ds['vmean_s']

    ds['error_div'] = (ds.divergence-ds.divergence_track)
    ds['error_vort'] = (ds.vorticity-ds.vorticity_track)
    ds['error_div_abs'] = np.fabs(ds.divergence-ds.divergence_track)
    ds['error_vort_abs'] = np.fabs(ds.vorticity-ds.vorticity_track)
    ds['error_div_sqr'] = (
        ds.divergence-ds.divergence_track)**2
    ds['error_vort_sqr'] = (
        ds.vorticity-ds.vorticity_track)**2
  #  ds['rmsvd_vort'] = np.sqrt(
   #     ds['error_vort_sqr'].sum()/ds['cos_weight'].sum())
   # ds['rmsvd_div'] = np.sqrt(ds['error_div_sqr'].sum()/ds['cos_weight'].sum())
    ds['rmsvd_vort'] = np.sqrt(ds['error_vort_sqr'].mean())
    ds['rmsvd_div'] = np.sqrt(ds['error_div_sqr'].mean())
   # ds['rmsvd_div'] = np.sqrt(ds['error_div_sqr'].sum()/ds['cos_weight'].sum())
   # ds['abs_div_weighted'] = np.fabs(ds['divergence'])*ds['cos_weight']
   # ds['abs_vort_weighted'] = np.fabs(ds['vorticity'])*ds['cos_weight']
   # ds['abs_vort_mean_weighed'] = ds['abs_vort_weighted'].sum() / \
    #    ds['cos_weight'].sum()
    # ds['abs_div_mean_weighed'] = ds['abs_div_weighted'].sum() / \
    #   ds['cos_weight'].sum()
    # ds['abs_div_mean_weighed'] = ds['abs_div_weighted'].sum() / \
    #   ds['cos_weight'].sum()
    ds['abs_vort_mean'] = np.fabs(ds['vorticity']).mean()
    ds['abs_div_mean'] = np.fabs(ds['divergence']).mean()
    return ds


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    flow = flow.astype(np.float32)
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC)
    return res


def ds_plotter(ds, flag):
    print(flag)
    print(ds)

    flag = PATH+flag
    quiver_plotter(ds, flag + 'groundt_s', 'umean_s', 'vmean_s')
    quiver_plotter(ds, flag + 'track_s', 'utrack_s', 'vtrack_s')
    quiver_plotter(ds, flag + 'groundt', 'umean', 'vmean')
    quiver_plotter(ds, flag + 'track', 'utrack', 'vtrack')
    quiver_plotter(ds, flag + 'tracke', 'u_error', 'v_error')
    quiver_plotter(ds, flag + 'tracke_s', 'u_error_s', 'v_error_s')

    map_plotter(ds, flag + 'vorticity_track',
                'vorticity_track', '1e4 m/s', -1, 1)
    map_plotter(ds, flag + 'vorticity', 'vorticity', '1e4 m/s', -1, 1)
    map_plotter(ds, flag + 'divergence_track',
                'divergence_track', '1e4 m/s', -1, 1)
    map_plotter(ds, flag + 'divergence', 'divergence', '1e4 m/s', -1, 1)
    map_plotter(ds, flag + 'error_div', 'error_div', '1e4 m/s', -1, 1)


def flow_calculator(ds):

    frame0 = np.squeeze(ds['vorticity_track'].values)
    frame0[abs(frame0) < 0.2] = np.nan
    mask0 = np.isnan(frame0)
    frame0 = np.nan_to_num(frame0)
    nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    frame = np.squeeze(ds['vorticity'].values)
    frame[abs(frame) < 0.2] = np.nan
    mask = np.isnan(frame)
    frame = np.nan_to_num(frame)
    nframe = cv2.normalize(src=frame, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    optical_flow = cv2.optflow.createOptFlow_DeepFlow()
    flowd = optical_flow.calc(nframe0, nframe, None)
    frame0d = warp_flow(frame0, flowd.copy())
    frame0d[mask] = np.nan
    #frame0d[abs(frame0d) < 0.2] = np.nan
    ds['vorticity_track'] = (['lat', 'lon'], frame0d)
    ds['error_vort_sqr'] = (
        ds.vorticity-ds.vorticity_track)**2
    ds['rmsvd_vort_flow'] = np.sqrt(ds['error_vort_sqr'].mean())
    return ds


def main():

    file = '../data/processed/experiments/900_700_full_november.nc'
    kernel = 3
    ds = xr.open_dataset(file)

    #ds_thresh = grad_calculator(ds.copy(), 200, 0.2)
    ds_all = grad_calculator(ds.copy(), 200, -1)

    # ds_plotter(ds_thresh, 'thresh')
#    ds_flow = flow_calculator(ds_all.copy())
    print('flow')
    print(ds_all)
    ds_plotter(ds_all, 'all')


if __name__ == "__main__":
    main()
