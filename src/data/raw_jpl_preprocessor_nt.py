import xarray as xr
from metpy.interpolate import interpolate_1d
import numpy as np
import glob
from pathlib import Path
from natsort import natsorted
import datetime


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = datetime.timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


d0 = datetime.datetime(2006, 7, 2, 23, 0, 0, 0)
d1 = datetime.datetime(2006, 7, 3, 19, 0, 0, 0)

start_dates = daterange(d0, d1, 6)
end_dates = daterange(d0+datetime.timedelta(hours=2),
                      d1+datetime.timedelta(hours=2), 6)
date_list = []
for i, start_date in enumerate(start_dates):
    date_list = date_list + daterange(start_date, end_dates[i], 0.5)

files = natsorted(
    glob.glob('../../data/raw/experiments/jpl/july/3/*.nc'))
ds_total = 0
for i, file in enumerate(files):
    print('var file:')
    print(file)
    ds = xr.open_dataset(file)
    ds = ds.expand_dims('time')
    date = np.array([date_list[i]])
    ds = ds.assign_coords(time=date)
    for var in ('u', 'v', 'qv'):
        ds[var].encoding['_FillValue'] = np.nan
        ds[var].encoding['missing_value'] = np.nan
    filename = Path(file).stem
    print(ds.time)
    if(i == 0):
        ds_total = ds
    else:
        ds_total = xr.concat([ds_total, ds], 'time')
print('saving..')
ds_total.to_netcdf('../../data/interim/experiments/july/3.nc')
print(ds_total)
