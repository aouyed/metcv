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


months = [7, 1]
pressures = [850, 500]
dts = ['30min', '60min']
for dt in dts:
    for month in months:
        for pressure in pressures:
            for day in (1, 2, 3):
                if month == 1:
                    month_str = 'january'
                else:
                    month_str = 'july'

                print(month_str)

                file = '../../data/interim/experiments/'+month_str + \
                    '/tracked/'+dt+'/combined/' + \
                    str(pressure)+'_'+month_str+'.nc'
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
                    qv = ds['qv'].sel(time=str(date)).values
                    print('calculating gradient ...')
                    grad = mpcalc.gradient(qv, deltas=(dy, dx))
                    grad = np.array([grad[0].magnitude, grad[1].magnitude])
                    grady = grad[0]
                    gradx = grad[1]
                    grad_mag = np.sqrt(gradx**2+grady**2)
                    print('building data arrays...')
                    da = build_datarray(gradx, lat, lon, date)
                    da1 = build_datarray(grady, lat, lon, date)
                    da2 = build_datarray(grad_mag, lat, lon, date)
                    ds_unit = xr.Dataset({'grad_x_qv': da})
                    ds_unit = xr.merge(
                        [ds_unit, xr.Dataset({'grad_y_qv': da1})])
                    ds_unit = xr.merge(
                        [ds_unit, xr.Dataset({'grad_mag_qv': da2})])
                    if len(ds_tot) > 0:
                        ds_tot = xr.concat([ds_tot, ds_unit], 'time')
                    else:
                        ds_tot = ds_unit
                print(ds_tot)
                ds_tot.to_netcdf(
                    '../../data/interim/experiments/'+month_str+'/tracked/'+dt+'/combined/'+str(pressure)+'_'+month_str+'_qv_grad.nc')
