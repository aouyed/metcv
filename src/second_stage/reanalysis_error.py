import pdb
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import datetime
from scipy.interpolate import LinearNDInterpolator as interpolator


def error_calc(df_0, pressure, month):
    """Calculates forecasting error from reanalysis."""
    pressure = str(pressure)
    month = str(month)
    print('era dataset')
    ds0 = xr.open_dataset(
        "../data/raw/experiments/reanalysis/era5/u_" + pressure + "_2006_0"+month+"_01_12:00:00_era5.nc")
    ds1 = xr.open_dataset(
        "../data/raw/experiments/reanalysis/era5/v_" + pressure + "_2006_0"+month+"_01_12:00:00_era5.nc")
    ds = xr.merge([ds0, ds1])
    print(ds)
    df = ds.to_dataframe()
    df = df.reset_index()
    df_era = df[['latitude', 'longitude', 'u', 'v']]
    df_era['longitude'] = df_era['longitude']-180

    print('cfs dataset')
    ds = xr.open_dataset(
        "../data/raw/experiments/reanalysis/cfs/pgbhnl.gdas.20060"+month+"0112.nc")
    df = ds.to_dataframe()
    df = df.reset_index()

    df_cfs = df[['latitude', 'longitude', 'UGRD_' +
        pressure+'mb', 'VGRD_'+pressure+'mb']]
  # df_cfs = df[['latitude', 'longitude', 'UGRD_850mb', 'VGRD_850mb']]
    df_cfs['longitude'] = df_cfs['longitude']-180

    # print(df_cfs)
    # print('merra dataset')
    # ds = xr.open_dataset("MERRA2_300.inst3_3d_asm_Np.20060101.nc4")
    # df = ds.to_dataframe()
    # df = df.reset_index()
    # date = datetime.datetime(2006, 1, 1, 12, 0, 0, 0)
    # df = df[(df.lev == 850) & (df.time == date)]
    # df = df.reset_index()
    # df_merra = df[['lat', 'lon', 'U', 'V']]

    # df_merra.columns = ['lat', 'lon', 'u', 'v']
    df_era.columns = ['lat', 'lon', 'u', 'v']
    df_cfs.columns = ['lat', 'lon', 'u', 'v']
    df_era = df_era.dropna()
    # df_merra = df_merra.dropna()
    df_cfs = df_cfs.dropna()

    dict_path = '../data/interim/dictionaries/dataframes.pkl'
    dataframes_dict = pickle.load(open(dict_path, 'rb'))

    df = df_0
    df = df.dropna()
    print('interpolating era data...')
    era_u_function = interpolator(
        points=df_era[['lat', 'lon']].values, values=df_era.u.values)
    era_v_function = interpolator(
        points=df_era[['lat', 'lon']].values, values=df_era.v.values)

#    print('interpolating merra data...')

 #   merra_u_function = interpolator(
  #      points=df_merra[['lat', 'lon']].values, values=df_merra.u.values)
  #  merra_v_function = interpolator(
   #     points=df_merra[['lat', 'lon']].values, values=df_merra.v.values)

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

    # df['u_merra'] = merra_u_function(
     #   df[['lat', 'lon']].values)

   # df['v_merra'] = merra_v_function(
    #    df[['lat', 'lon']].values)

    df['u_cfs'] = cfs_u_function(
        df[['lat', 'lon']].values)

    df['v_cfs'] = cfs_v_function(
        df[['lat', 'lon']].values)

    df[['u_era', 'v_era',  'u_cfs', 'v_cfs']] = df[[
        'u_era', 'v_era', 'u_cfs', 'v_cfs']].fillna(0)
    df['u_error_rean'] = df['u_cfs']-df['u_era']
    df['v_error_rean'] = df['v_cfs']-df['v_era']
    df['error_mag'] = np.sqrt(df['u_error_rean']**2 + df['v_error_rean']**2)
    return df
\
