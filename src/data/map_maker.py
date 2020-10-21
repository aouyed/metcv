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

    if dt == 3600:
        path_jpl = path_jpl_60min
    elif dt == 1800:
        path_jpl = path_jpl_30min
    else:
        raise ValueError('not supported value in dt')

    df_dict = {}
    for pressure_i in (850, 500):
        for dt in (1800, 3600):
            for month in ('january', 'july'):
                df = pd.read_pickle(bp.PATH_DF+str(dt)+'_'+month+'_' +
                                    str(pressure_i)+'_df_results.pkl')
                df_dict[(dt, month, pressure_i)] = df

    edp.multiple_filter_plotter(df_dict, bp.PATH_PLOT+'january' +
                                '_results_test', 'january')
    edp.multiple_filter_plotter(df_dict, bp.PATH_PLOT+'july' +
                                '_results_test', 'july')

    ds = xr.open_dataset(PATH+str(dt)+'_'+str(pressure)+'_'+month+'.nc')
    ds = ds.sel(time=str(ds.time.values[0]))
    ds = ds_error(ds)

    ds_full = xr.open_dataset(
        PATH+str(dt)+'_'+str(pressure)+'_full_'+month+'.nc')
    ds_full = ds_full.sel(time=str(ds_full.time.values[0]))
    ds_full = ds_error(ds_full)

    ds_jpl = xr.open_dataset(path_jpl+str(pressure) + '_'+month+'.nc')
    ds_jpl = ds_jpl.sel(time=str(ds_jpl.time.values[0]))
    ds_jpl = ds_error(ds_jpl, rean=False)

    plotter(ds, ds_full, ds_jpl,  'error_mag',
            month+'_'+str(dt), pressure)


if __name__ == "__main__":
    main()
