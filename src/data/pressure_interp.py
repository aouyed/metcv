import xarray as xr
from metpy.interpolate import interpolate_1d
import numpy as np
import glob
from pathlib import Path
from natsort import natsorted


varnames = ['u', 'v', 'qv']

for varname in varnames:

    files = natsorted(
        glob.glob('../../data/raw/experiments/07_01_2006/' + varname+'/netcdf/*.nc4'))
    files_pl = natsorted(
        glob.glob('../../data/raw/experiments/07_01_2006/pl/*.nc4*'))

    path = '../../data/raw/experiments/07_01_2006/'+varname+'/850/'
    files_inter = natsorted(glob.glob(
        '../../data/raw/experiments/07_01_2006/'+varname+'/850/*.npy'))

    da = 0
    for i, file in enumerate(files):
        # if i > 2:
         #   break
        print('var file:')
        print(file)
        print('pl file')
        print(files_pl[i])
        ds = xr.open_dataset(file)
        var = ds[varname.upper()]
        var = var.values
        ds = xr.open_dataset(files_pl[i])

        pl = ds['PL']
        pl = pl.values/100

        print('interpolating')

        #var_interp = np.load(files_inter[i])
        var_interp = interpolate_1d(850, pl, var, axis=1)
        var_interp = np.squeeze(var_interp)

        if(i == 0):
            da = xr.DataArray(var_interp, coords=[
                              ds.lat.values, ds.lon.values], dims=['lat', 'lon'])
            da = da.expand_dims('time')
            da = da.assign_coords(time=ds.time.values)
        else:
            da_unit = xr.DataArray(var_interp, coords=[
                                   ds.lat.values, ds.lon.values], dims=['lat', 'lon'])
            da_unit = da_unit.expand_dims('time')
            da_unit = da_unit.assign_coords(time=ds.time.values)
            #import pdb
            # pdb.set_trace()
            da = xr.concat([da, da_unit], 'time')
        filename = Path(file).stem

        #np.save(path+filename, var_interp)
        print('done interpolation')
    ds = xr.Dataset({varname: da})
    ds.to_netcdf('../../data/interim/netcdf/07_01_2006/'+varname+'.nc')
    print(ds)
