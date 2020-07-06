import os
import xarray as xr
import glob
import sh
PATH = '../data/processed/experiments/'


def preprocessing_loop(exp_filters, name):
    ds_total = xr.Dataset()
    for i, exp_filter in enumerate(exp_filters):
        files = glob.glob('../data/processed/experiments/'+exp_filter+'*')
        ds_con = xr.Dataset()
        for j, file in enumerate(files):
            print('var file:')
            print(file)
            ds = xr.open_dataset(file)
            ds = ds.expand_dims('filter')
            ds = ds.assign_coords(filter=[exp_filter])
            if(j == 0):
                ds_con = ds
            else:
                ds_con = xr.concat([ds_con, ds], 'time')
        if(i == 0):
            ds_total = ds_con
        else:
            ds_total = xr.concat([ds_total, ds_con], 'filter')
        if files:
            sh.rm(files)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    ds_total = ds_total.rename({'umeanh': 'umean', 'vmeanh': 'vmean'})
    ds_total.to_netcdf(PATH + name+'.nc',  mode='w')


def run(pressure):
    exp_filters = ('exp2', 'ground_t', 'df')
    preprocessing_loop(exp_filters, str(pressure) + '_july')
    preprocessing_loop(['full_exp2'], str(pressure) + '_full_july')
