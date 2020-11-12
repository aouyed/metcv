import xarray as xr
from datetime import datetime
from data import extra_data_plotter as edp
import numpy as np
from data import batch_plotter as bp
import pandas as pd
VMAX = 12
PATH = '../data/processed/experiments/'


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
    edp.map_plotter_multiple(ds, ds_full, ds_jpl, str(dt)+'_'+str(pressure) + '_' +
                             varname, 'm/s', 0, vmax)


def main(triplet, pressure=850, dt=3600):

    month = triplet.strftime("%B").lower()
    path_jpl_60min = '../data/interim/experiments/'+month+'/tracked/60min/combined/'
    path_jpl_30min = '../data/interim/experiments/'+month+'/tracked/30min/combined/'

    df = pd.read_pickle(bp.PATH_DF+str(dt)+'_'+month+'_' +
                        str(pressure)+'_df_results.pkl')
    edp.filter_plotter(df, bp.PATH_PLOT+month+'_'+str(dt)+'_'+str(pressure) +
                       '_results_test', ' ')


if __name__ == "__main__":
    main()
