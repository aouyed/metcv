import xarray as xr
from datetime import datetime
import extra_data_plotter as edp
import numpy as np

VMAX = 16.748113280632353


def ds_averager(ds, rean=True):
    ds_total = xr.Dataset()
    for date in ds.time.values:
        print('averaging date: '+str(date))
        ds_unit = ds.sel(time=date)
        if ds_total:
            ds_total += ds_unit
        else:
            ds_total = ds_unit
    ds_average = ds_total/ds.time.values.size
    ds_average['v_error'] = ds_average['vtrack']-ds_average['vmean']
    ds_average['u_error'] = ds_average['utrack']-ds_average['umean']
    ds_average['error_mag'] = np.sqrt(
        ds_average['v_error']**2+ds_average['u_error']**2)
    if rean:
        ds_average['error_mag_rean'] = np.sqrt(
            ds_average['v_error_rean']**2+ds_average['u_error_rean']**2)

    return ds_average


def plotter(ds, varname, filter):

    var = ds[varname].values
    vmin = 0
    #vmax = np.quantile(np.nan_to_num(var), 0.99)
    vmax = VMAX
    edp.map_plotter(var, varname + '_' + filter, 'm/s', 0, vmax)


ds = xr.open_dataset('../../data/processed/experiments/july.nc')
filter = 'df'
ds = ds.sel(filter=filter)
ds = ds_averager(ds)
plotter(ds, 'error_mag', filter)

ds = xr.open_dataset('../../data/processed/experiments/full_july.nc')
filter = 'full_exp2'
ds = ds.sel(filter=filter)
ds = ds_averager(ds)
plotter(ds, 'error_mag', filter)

ds = xr.open_dataset('../../data/processed/experiments/full_july.nc')
filter = 'full_exp2'
ds = ds.sel(filter=filter)
ds = ds_averager(ds)
plotter(ds, 'error_mag_rean', filter)

ds = xr.open_dataset(
    '../../data/interim/experiments/july/tracked/60min/combined/july.nc')
filter = 'jpl'
ds = ds_averager(ds, rean=False)
plotter(ds, 'error_mag', filter)
