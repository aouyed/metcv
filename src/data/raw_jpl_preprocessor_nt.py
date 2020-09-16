import xarray as xr\
    from metpy.interpolate import interpolate_1d
import numpy as np
import glob
from pathlib import Path
from natsort import natsorted
import datetime
from dateutil.parser import parse


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = datetime.timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


def date_test(file, date):
    file_string = Path(file).stem
    file_string = file_string.split('_')

    date_file = file_string[-2] + file_string[-1]
    date_file = parse(date_file)
    assert date == date_file
    print('succesful test!')


months = [1, 7]
dts = ['30min', '60min']
days = [1, 2, 3]
for month in months:
    if month == 1:
        month_str = 'january'
    else:
        month_str = 'july'
    for day in days:
        d1 = datetime.datetime(2006, month, day, 19, 0, 0, 0)
        d0 = d1 - datetime.timedelta(hours=20)
        print('d0 ' + str(d0))
        start_dates = daterange(d0, d1, 6)
        end_dates = daterange(d0+datetime.timedelta(hours=2),
                              d1+datetime.timedelta(hours=2), 6)
        date_list = []
        for i, start_date in enumerate(start_dates):
            date_list = date_list + daterange(start_date, end_dates[i], 0.5)
        day = str(day)
        files = natsorted(
            glob.glob('../../data/raw/experiments/jpl/'+month_str+'/0'+day+'/*.nc'))
        ds_total = xr.Dataset()
        for i, file in enumerate(files):
            print('var file:')
            print(file)
            ds = xr.open_dataset(file)
            ds = ds.expand_dims('time')
            date = np.array([date_list[i]])
            date_test(file, date)
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
        ds_total.to_netcdf('../../data/interim/experiments/july/'+day+'.nc')
        print(ds_total)
