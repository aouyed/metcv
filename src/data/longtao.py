import xarray as xr
import netCDF4
import glob
import numpy as np
filenames = glob.glob("../../data/raw/jpl/processed_jpl/*")
print(filenames)

ds = xr.open_dataset(filenames[0])
print(ds)
#print(ds)
#T = ds.sel(pressure=850, method='nearest')
#print(T)

