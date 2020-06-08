import xarray as xr
import glob


def run():
    exp_filters = ('exp2', 'ground_t')

    ds_total = xr.Dataset()
    for i, exp_filter in enumerate(exp_filters):
        files = glob.glob('../data/processed/experiments/'+exp_filter+'*')
        ds_con = xr.Dataset()
        for j, file in enumerate(files):
            print('var file:')
            print(file)
            ds = xr.open_dataset(file)
            print(ds)
            ds = ds.expand_dims('filter')
            ds = ds.assign_coords(filter=[exp_filter])
            if(j == 0):
                ds_con = ds
            else:
                ds_con = xr.concat([ds_con, ds], 'time')
        print(ds_con)
        if(i == 0):
            ds_total = ds_con
        else:
            ds_total = xr.concat([ds_total, ds_con], 'filter')
    ds_total.to_netcdf('../data/processed/experiments/july.nc')
