import xarray as xr
from datetime import datetime
import extra_data_plotter as edp
import numpy as np

ds = xr.open_dataset('../../data/processed/experiments/july.nc')
ds_track = xr.open_dataset(
    '../../data/interim/experiments/july/tracked/60min/combined/july.nc')

ds['v_error'] = ds['vtrack']-ds['vmeanh']
ds['u_error'] = ds['utrack']-ds['umeanh']
ds['error_mag'] = np.sqrt(ds['v_error']**2+ds['u_error']**2)

date = datetime(2006, 7, 1, 0, 0, 0, 0)

ds_unit = ds.sel(time=date, filter='df')
var = ds_unit['error_mag'].values
vmin = 0
vmax = np.quantile(np.nan_to_num(var), 0.99)
vmax = 5.838104948848125
print(vmax)
edp.map_plotter(var, 'errorf', 'm/s', 0, vmax)
