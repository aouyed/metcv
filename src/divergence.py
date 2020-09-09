import xarray as xr
import pdb
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import map_maker as mm
from data import extra_data_plotter as edp
COORDS = [(-30, 30), (-60, -30), (-90, -60), (30, 60), (60, 90)]


def plotter(ds, varname, dt, pressure, filter):

    var = ds[varname].values
    var = np.squeeze(var)
    print('test')
    print(np.mean(np.nan_to_num(var)))
    vmin = 0
    vmax = 0.001
    edp.map_plotter(var, dt+'_'+str(pressure) + '_' +
                    varname + '_' + filter, 'm/s', 0, vmax)


def filter_plotter(df0, values, title):
    fig, ax = plt.subplots()
    filters = ['vorticity_rmse', 'vorticity_mean',
               'divergence_rmse', 'divergence_mean']

    for filter in filters:
        df = df0[df0.exp_filter == filter]
        ax.plot(df['latlon'], df['rmse'], '-o', label=filter)

    ax.legend(frameon=None)
    #ax.set_ylim(0, ERROR_MAX)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSE [1/s]")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)
    plt.close()


def build_datarray(data, lat, lon, date):
    da = xr.DataArray(data, coords=[
        lat, lon], dims=['lat', 'lon'])
    da = da.expand_dims('time')
    da = da.assign_coords(time=[date])
    return da


def grad_calculator(ds, dy, dx, date):
    u = ds['umean'].sel(time=str(date)).values
    v = ds['vmean'].sel(time=str(date)).values
    u = np.squeeze(u)
    v = np.squeeze(v)
    div = div_calc(u, v, dx, dy)
    vort = vort_calc(u, v, dx, dy)
    print('building data arrays...')
    da3 = build_datarray(div, lat, lon, date)
    da4 = build_datarray(vort, lat, lon, date)
    ds_unit = xr.Dataset({'divergence': da3})
    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'vorticity': da4})])

    u = ds['utrack'].sel(time=str(date)).values
    v = ds['vtrack'].sel(time=str(date)).values
    u = np.squeeze(u)
    v = np.squeeze(v)

    div = div_calc(u, v, dx, dy)
    vort = vort_calc(u, v, dx, dy)
    da3 = build_datarray(div, lat, lon, date)
    da4 = build_datarray(vort, lat, lon, date)
    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'divergence_track': da3})])
    ds_unit = xr.merge(
        [ds_unit, xr.Dataset({'vorticity_track': da4})])
    ds_unit['cos_weight'] = ds['cos_weight'].sel(time=str(date))
    ds_unit['error_div'] = abs(ds_unit.divergence-ds_unit.divergence_track)
    ds_unit['error_vort'] = abs(ds_unit.vorticity-ds_unit.vorticity_track)
    print(ds_unit)
    return ds_unit


def div_calc(u, v, dx, dy):
    div = mpcalc.divergence(
        u * units['m/s'], v * units['m/s'], dx, dy, dim_order='yx')
    div = div.magnitude
    #div = np.nan_to_num(div)
    return div


def vort_calc(u, v, dx, dy):
    vort = mpcalc.vorticity(
        u * units['m/s'], v * units['m/s'], dx, dy, dim_order='yx')
    vort = vort.magnitude
    #vort = np.nan_to_num(vort)

    return vort


def coord_to_string(coord):
    if coord[0] < 0:
        lowlat = str(abs(coord[0])) + '°S'
    else:
        lowlat = str(coord[0]) + '°N'

    if coord[1] < 0:
        uplat = str(abs(coord[1])) + '°S'
    else:
        uplat = str(coord[1]) + '°N'
    stringd = str(str(lowlat)+',' + str(uplat))
    return stringd


def rmse_lists(df, rmses, region, filter_res):
    for coord in COORDS:

        rmse_div, rmse_vort, mean_div, mean_vort = rmse_calculator(
            df, coord)

        stringc = coord_to_string(coord)

        filter_res.append('divergence_rmse')
        rmses.append(rmse_div)
        region.append(stringc)
        filter_res.append('vorticity_rmse')
        rmses.append(rmse_vort)
        region.append(stringc)
        filter_res.append('divergence_mean')
        rmses.append(mean_div)
        region.append(stringc)
        filter_res.append('vorticity_mean')
        rmses.append(mean_vort)
        region.append(stringc)


def rmse_calculator(df, coord):
    df_unit = df[(df.lat >= coord[0]) & (df.lat <= coord[1])]
    df_unit['delta_div'] = df_unit.cos_weight * \
        (df_unit.divergence-df_unit.divergence_track)**2
    df_unit['delta_vort'] = df_unit.cos_weight * \
        (df_unit.vorticity-df_unit.vorticity_track)**2
    df_unit['weighed_abs_div'] = df_unit.cos_weight*abs(df_unit.divergence)
    df_unit['weighed_abs_vort'] = df_unit.cos_weight*abs(df_unit.vorticity)

    rmse_div = np.sqrt(df_unit['delta_div'].sum()/df_unit['cos_weight'].sum())
    rmse_vort = np.sqrt(df_unit['delta_vort'].sum() /
                        df_unit['cos_weight'].sum())
    mean_div = df_unit['weighed_abs_div'].sum()/df_unit['cos_weight'].sum()
    mean_vort = df_unit['weighed_abs_vort'].sum()/df_unit['cos_weight'].sum()
    return rmse_div, rmse_vort, mean_div, mean_vort


months = [7]
pressures = [850, 500]
dts = [3600]
rmses = []
region = []
filter_res = []
file_name = '3600_850_full_july.nc'

file = '../data/processed/experiments/'+file_name
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
    ds_unit = grad_calculator(ds, dy, dx, date)
    if len(ds_tot) > 0:
        ds_tot = xr.concat([ds_tot, ds_unit], 'time')
    else:
        ds_tot = ds_unit
df = ds_tot.to_dataframe().dropna().reset_index()
rmse_lists(df, rmses, region, filter_res)
d = {'latlon': region, 'exp_filter': filter_res, 'rmse': rmses}
df_results = pd.DataFrame(data=d)
print(df_results)
filter_plotter(df_results, 'div_vort', ' ')
ds_tot = ds_tot.sel(time=str(ds_tot.time.values[0]))
ds_tot['divergence'] = abs(ds_tot.divergence)
ds_tot['vorticity'] = abs(ds_tot.vorticity)
plotter(ds_tot, 'error_div', '850', '850', 'div')
plotter(ds_tot, 'error_vort', '850', '850', 'vort')
plotter(ds_tot, 'divergence', '850', '850', 'divergence')
plotter(ds_tot, 'vorticity', '850', '850', 'vorticity')
