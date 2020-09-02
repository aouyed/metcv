import xarray as xr
from metpy.interpolate import interpolate_1d
import numpy as np
import glob
from pathlib import Path
from natsort import natsorted
from datetime import datetime
import sh

PRESSURES = (500, 850)


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = datetime.timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


print('hello')
months = [1, 7]
dts = ['30min', '60min']
for dt in dts:
    for month in months:
        for pressure in PRESSURES:
            if month == 1:
                month_str = 'january'
            else:
                month_str = 'july'
            path_files = '../../data/interim/experiments/'+month_str+'/tracked/'+dt+'/*.nc'
            rm_files = natsorted(glob.glob(path_files))

            for day in (1, 2, 3):

                date_list = (datetime(2006, month, day, 0, 0, 0, 0), datetime(2006, month, day, 6, 0, 0, 0),
                             datetime(2006, month, day, 12, 0, 0, 0), datetime(2006, month, day, 18, 0, 0, 0))

                files = natsorted(
                    glob.glob('../../data/raw/experiments/jpl/tracked/'+month_str+'/'+str(day)+'/'+str(pressure)+'/'+dt+'/*.nc'))
                print(files)
                ds_total = xr.Dataset()
                for i, file in enumerate(files):
                    print('var file:')
                    print(file)
                    ds = xr.open_dataset(file)
                    ds = ds.expand_dims('time')
                    date = np.array([date_list[i]])
                    ds = ds.assign_coords(time=date)
                    for var in ds:
                        print('var', var)
                        ds[var].encoding['_FillValue'] = np.nan
                        ds[var].encoding['missing_value'] = np.nan
                    filename = Path(file).stem
                    print(ds.time)
                    if not ds_total:
                        ds_total = ds
                    else:
                        ds_total = xr.concat([ds_total, ds], 'time')
                print('saving..')
                ds_total.to_netcdf(
                    '../../data/interim/experiments/'+month_str+'/tracked/'+dt+'/' + str(day)+'.nc')
                print(ds_total)

            ds_total = xr.Dataset()

            files = natsorted(glob.glob(path_files))

            for file in files:
                ds = xr.open_dataset(file)
                if not ds_total:
                    ds_total = ds
                else:
                    ds_total = xr.concat([ds_total, ds], 'time')
                file_path = '../../data/interim/experiments/'+month_str + \
                    '/tracked/'+dt+'/combined/' + \
                    str(pressure)+'_'+month_str+'.nc'
                sh.rm(file_path)
                ds_total.to_netcdf(file_path)
                print(ds_total)
