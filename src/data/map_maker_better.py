import xarray as xr
from datetime import datetime
from data import extra_data_plotter as edp
import numpy as np
from data import batch_plotter as bp
import pandas as pd
VMAX = 12
PATH = '../data/processed/experiments/'


def ds_error(ds, rean=True):
    ds['v_error'] = ds['vtrack']-ds['vmean']
    ds['u_error'] = ds['utrack']-ds['umean']

    ds['error_mag'] = np.sqrt(
        ds['v_error']**2+ds['u_error']**2)
    if rean:
        ds['error_mag_rean'] = np.sqrt(
            ds['v_error_rean']**2+ds['u_error_rean']**2)

    return ds


def plotter(ds, ds_full, ds_jpl,  varname, dt, pressure):

    vmin = 0
    vmax = VMAX
    edp.map_plotter_multiple_better(ds, ds_full, ds_jpl, str(dt)+'_'+str(pressure) + '_' +
                                    varname, 'm/s', 0, vmax)


def main(triplet, pressure=850, dt=3600):

    month = triplet.strftime("%B").lower()

    ds = xr.open_dataset(PATH+str(dt)+'_'+str(pressure)+'_'+month+'.nc')
    ds = ds.sel(time=str(ds.time.values[0]))
    ds = ds_error(ds)

    ds_full = xr.open_dataset(
        PATH+str(dt)+'_'+str(pressure)+'_full_'+month+'.nc')
    ds_full = ds_full.sel(time=str(ds_full.time.values[0]))
    ds_full = ds_error(ds_full)

    ds_jpl = xr.zeros_like(ds_full.loc[{'filter': 'full_exp2'}].copy())
    #ds_jpl = ds_jpl.sel(time=str(ds_jpl.time.values[0]))
    ds_jpl = ds_error(ds_jpl, rean=False)

    plotter(ds, ds_full, ds_jpl,  'error_mag',
            month+'_'+str(dt), pressure)


if __name__ == "__main__":
    main()
