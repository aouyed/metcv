import pdb
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import numpy as np
from tqdm import tqdm
import pickle
import glob
from natsort import natsorted
import pandas as pd


def rmsvd_calculator(df, coord, rmsvd_num, rmsvd_den, filter):
    df_unit = df[(df.lat >= coord[0]) & (df.lat <= coord[1])]
    df_unit = df_unit[df_unit['filter'] == filter]
    df_unit['cos_weight'] = np.cos(df_unit.lat/180*np.pi)
    erroru = df_unit.utrack-df_unit.umeanh
    errorv = df_unit.vtrack-df_unit.vmeanh
    df_unit['vec_diff'] = df_unit.cos_weight * (erroru**2 + errorv**2)
    rmsvd_num = rmsvd_num + df_unit['vec_diff'].sum()
    rmsvd_den = rmsvd_den + df_unit['cos_weight'].sum()
    return rmsvd_num, rmsvd_den


def df_builder(ds, ds_track, date):
    ds = ds.sel(time=date)
    ds_track = ds_track.sel(time=date)
    #ds_track = ds_track.expand_dims('filter')
    #ds_track = ds_track.assign_coords(filter=['jpl'])
    df = ds.to_dataframe().reset_index()
    dft = ds_track.to_dataframe().reset_index()
    df_tot = df.merge(dft.dropna(), on=[
        'lat', 'lon'], how='left', indicator='Exist')
    #df_dummy = df_tot[df_tot['filter'] != 'ground_t'].copy()
    df_tot = df_tot[df_tot.utrack_y.notna()]
    df_tot = df_tot.drop(
        columns=['Exist', 'time_x', 'time_y', 'utrack_y'])
    df_tot = df_tot.rename(columns={'utrack_x': 'utrack'})
    #df_tot = df_tot.dropna()

    return df_tot


def plot_preprocessor(ds, ds_track):
    dates = ds.time.values
    coords = [(-30, 30), (-60, -30), (-90, -60), (30, 60), (60, 90)]
    #coords = [(-90, 90)]
    filters = ['df', 'exp2', 'ground_t']
    rmsvds = []
    region = []
    filter_res = []
    print('preprocessing_data...')

    for i, date in enumerate(dates):
        df_unit = df_builder(ds, ds_track[[
            'time', 'utrack', 'lat', 'lon']], date)
        df_unit.to_pickle('../data/interim/experiments/dataframes/' +
                          str(i)+'.pkl')

    files = natsorted(glob.glob('../data/interim/experiments/dataframes/*'))
    for coord in tqdm(coords):
        for filter in filters:
            rmsvd_num = 0
            rmsvd_den = 0
            for i, date in enumerate(dates):
                df_unit = pd.read_pickle(files[i])
                df_unit = df_unit.dropna()
                rmsvd_num, rmsvd_den = rmsvd_calculator(
                    df_unit, coord, rmsvd_num, rmsvd_den, filter)
               # if(filter == 'ground_t'):
                #    pdb.set_trace()

            rmsvds.append(np.sqrt(rmsvd_num/rmsvd_den))
            region.append(coord)
            filter_res.append(filter)

    d = {'latlon': region, 'categories': filter_res,
         'rmse': rmsvds, 'exp_filter': filter_res}
    df_results = pd.DataFrame(data=d)
    return df_results


file = '../data/processed/experiments/july.nc'
ds = xr.open_dataset(file)
ds_track = xr.open_dataset(
    '../data/interim/experiments/july/tracked/60min/1.nc')

df = plot_preprocessor(ds, ds_track)
print(df)
pdb.set_trace()


fig, ax = plt.subplots()

df = df0[(df0.categories == 'rf') & (df0.exp_filter == 'exp2')]
ax.plot(df['latlon'], df['rmse'], '-o', label='UA (RF+VEM)')

df = df0[(df0.categories == 'ground_t') & (df0.exp_filter == 'ground_t')]
ax.plot(df['latlon'], df['rmse'], '-o',
        label='noisy observations')

df = df0[df0.categories == 'df']
ax.plot(df['latlon'], df['rmse'], '-o', label='VEM')

df = df0[df0.categories == 'jpl']
ax.plot(df['latlon'], df['rmse'], '-o', label='JPL')

ax.legend(frameon=None)
ax.set_ylim(0, 5)
ax.set_xlabel("Region")
ax.set_ylabel("RMSVD [m/s]")
ax.set_title(title)
directory = '../data/processed/density_plots'
plt.savefig(values+'.png', bbox_inches='tight', dpi=300)
