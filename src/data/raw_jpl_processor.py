import xarray as xr
from metpy.interpolate import interpolate_1d
import numpy as np
import glob
from pathlib import Path
from natsort import natsorted
from datetime import datetime
import sh

PRESSURES = (500, 850)
PATH_FILES = '../../data/interim/experiments/january/tracked/30min/*.nc'


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = datetime.timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


rm_files = natsorted(glob.glob(PATH_FILES))

if rm_files:
    sh.rm(rm_files)

for pressure in PRESSURES:
    for day in (1, 2, 3):

        date_list = (datetime(2006, 1, day, 0, 0, 0, 0), datetime(2006, 1, day, 6, 0, 0, 0),
                     datetime(2006, 1, day, 12, 0, 0, 0), datetime(2006, 1, day, 18, 0, 0, 0))

        files = natsorted(
            glob.glob('../../data/raw/experiments/jpl/tracked/january/'+str(day)+'/'+str(pressure)+'/30min/*.nc'))
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
            '../../data/interim/experiments/january/tracked/30min/' + str(day)+'.nc')
        print(ds_total)

    ds_total = xr.Dataset()
    files = natsorted(glob.glob(PATH_FILES))

    for file in files:
        ds = xr.open_dataset(file)
        if not ds_total:
            ds_total = ds
        else:
            ds_total = xr.concat([ds_total, ds], 'time')
        ds_total.to_netcdf(
            '../../data/interim/experiments/january/tracked/30min/combined/'+str(pressure)+'_january.nc')
        print(ds_total)
