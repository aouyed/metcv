import xarray as xr
import pdb
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime
import numpy as np
PATH = '../data/processed/experiments/'


def main(triplet, alg,  pressure=500, dt=3600):

    hist_dict = {}
    month = triplet.strftime("%B").lower()

    ds_name = str(dt)+'_' + str(pressure) + '_' + \
        triplet.strftime("%B").lower()

    file = PATH + ds_name+'.nc'
    print(file)
    ds = xr.open_dataset(file)
    ds_raw = xr.open_dataset('../data/interim/experiments/november/4.nc')
    print(ds)
    print(ds_raw)
    date = ds.time.values[0]
    ds_raw = ds_raw.loc[{'pressure': pressure}].loc[{
        'time': str(date)}].drop('time').drop('pressure')
    ds = xr.merge([ds, ds_raw])
    print(ds)
    lat = ds.lat.values
    lon = ds.lon.values
    print('calculating deltas...')
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

    qv = np.squeeze(ds['qv'].values)
    print('calculating gradient ...')
    grad = mpcalc.gradient(qv, deltas=(dy, dx))
    grad = np.array([grad[0].magnitude, grad[1].magnitude])
    grady = grad[0]
    gradx = grad[1]
    grad_mag = np.sqrt(gradx**2+grady**2)
    print('building data arrays...')
    ds['grad_y_qv'] = (['lat', 'lon'], grady/1000)
    ds['grad_x_qv'] = (['lat', 'lon'], gradx/1000)
    ds['grad_mag_qv'] = (['lat', 'lon'], grad_mag/1000)
    ds['qv'] = ds['qv']/1000
    ds = xr.concat([ds, xr.zeros_like(ds.loc[{'filter': 'exp2'}]).assign_coords(
        {'filter': 'jpl'})], 'filter')
    ds = ds.astype(np.float32)
    ds.to_netcdf(PATH+ds_name+'_merged.nc')
