import xarray as xr
import netCDF4
import glob
import numpy as np
filenames = glob.glob("../../data/raw/july/*")
print(filenames)

ds = xr.open_dataset(filenames[2])
print(ds.time)
#print(ds)
#T = ds.sel(pressure=850, method='nearest')
#print(T)

