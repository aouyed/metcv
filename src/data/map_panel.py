import xarray as xr
from data import map_maker as mm

VMAX = 12
PATH = '../data/processed/experiments/'


def main(triplet, pressure=850, dt=3600):
    month = triplet.strftime("%B").lower()
    path_jpl_60min = '../data/interim/experiments/'+month+'/tracked/60min/combined/'
    path_jpl_30min = '../data/interim/experiments/'+month+'/tracked/30min/combined/'

    if dt == 3600:
        path_jpl = path_jpl_60min
    elif dt == 1800:
        path_jpl = path_jpl_30min
    else:
        raise ValueError('not supported value in dt')

    ds = xr.open_dataset(PATH+str(dt)+'_'+str(pressure)+'_'+month+'.nc')
    ds = ds.sel(time=str(ds.time.values[0]))
    ds = mm.ds_error(ds)

    ds_full = xr.open_dataset(
        PATH+str(dt)+'_'+str(pressure)+'_full_'+month+'.nc')
    ds_full = ds_full.sel(time=str(ds_full.time.values[0]))
    ds_full = mm.ds_error(ds_full)

    ds_jpl = xr.open_dataset(path_jpl+str(pressure) + '_'+month+'.nc')
    ds_jpl = ds_jpl.sel(time=str(ds_jpl.time.values[0]))
    ds_jpl = mm.ds_error(ds_jpl, rean=False)

    mm.plotter(ds, ds_full, ds_jpl,  'error_mag',
               month+'_'+str(dt), pressure)


if __name__ == "__main__":
    main()
