import pdb
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import datetime
from viz import amv_analysis as aa
from scipy.interpolate import LinearNDInterpolator as interpolator
import extra_data_plotter as edp

print('era dataset')
ds0 = xr.open_dataset("u_era5.nc")
ds1 = xr.open_dataset("v_era5.nc")
ds = xr.merge([ds0, ds1])
print(ds)
df = ds.to_dataframe()
df = df.reset_index()
df_era = df[['latitude', 'longitude', 'u', 'v']]
df_era['longitude'] = df_era['longitude']-180

print('cfs dataset')
ds = xr.open_dataset("cfs.nc")
print(ds)
df = ds.to_dataframe()
df = df.reset_index()

df_cfs = df[['latitude', 'longitude', 'UGRD_850mb', 'VGRD_850mb']]
df_cfs['longitude'] = df_cfs['longitude']-180

print(df_cfs)
print('merra dataset')
ds = xr.open_dataset("MERRA2_300.inst3_3d_asm_Np.20060101.nc4")
print(ds)
df = ds.to_dataframe()
df = df.reset_index()
date = datetime.datetime(2006, 1, 1, 12, 0, 0, 0)
df = df[(df.lev == 850) & (df.time == date)]
df = df.reset_index()
df_merra = df[['lat', 'lon', 'U', 'V']]

df_merra.columns = ['lat', 'lon', 'u', 'v']
df_era.columns = ['lat', 'lon', 'u', 'v']
df_cfs.columns = ['lat', 'lon', 'u', 'v']
df_era = df_era.dropna()
df_merra = df_merra.dropna()
df_cfs = df_cfs.dropna()

dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


start_date = datetime.datetime(2006, 7, 1, 6, 0, 0, 0)
end_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)
df = aa.df_concatenator(dataframes_dict, start_date,
                        end_date, False, True, False)

df = df.dropna()
print('interpolating era data...')
era_u_function = interpolator(
    points=df_era[['lat', 'lon']].values, values=df_era.u.values)
era_v_function = interpolator(
    points=df_era[['lat', 'lon']].values, values=df_era.v.values)

print('interpolating merra data...')

merra_u_function = interpolator(
    points=df_merra[['lat', 'lon']].values, values=df_merra.u.values)
merra_v_function = interpolator(
    points=df_merra[['lat', 'lon']].values, values=df_merra.v.values)


print('interpolating cfs data...')

cfs_u_function = interpolator(
    points=df_cfs[['lat', 'lon']].values, values=df_cfs.u.values)
cfs_v_function = interpolator(
    points=df_cfs[['lat', 'lon']].values, values=df_cfs.v.values)


print('regridding...')
df['u_era'] = era_u_function(
    df[['lat', 'lon']].values)

df['v_era'] = era_v_function(
    df[['lat', 'lon']].values)

df['u_merra'] = merra_u_function(
    df[['lat', 'lon']].values)

df['v_merra'] = merra_v_function(
    df[['lat', 'lon']].values)

df['u_cfs'] = cfs_u_function(
    df[['lat', 'lon']].values)

df['v_cfs'] = cfs_v_function(
    df[['lat', 'lon']].values)

df['u_error_rean'] = df['u_cfs']-df['u_era']
df['v_error_rean'] = df['v_cfs']-df['v_era']
df['error_mag'] = np.sqrt(df['u_error_rean']**2 + df['v_error_rean']**2)
print('mean error u')
print(abs(df['u_error_rean']).mean())
print('mean u')
print(df['u_era'].mean())
print('plotting...')
edp.map_plotter(df, 'error_mag', 'error_mag', 'm/s', 0, 10)
edp.quiver_plotter(df, 'era_quiver', 'u_era', 'v_era')
edp.quiver_plotter(df, 'cfs_quiver', 'u_cfs', 'v_cfs')
edp.quiver_plotter(df, 'merra_quiver', 'u_merra', 'v_merra')
edp.quiver_plotter(df, 'error_rean_quiver', 'u_error_rean', 'v_error_rean')
print('Done!')

pdb.set_trace()
