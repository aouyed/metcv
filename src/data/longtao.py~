import xarray as xr
import netCDF4
import glob 
filenames = glob.glob("../../data/raw/longtao_amv/raw/*")
print(filenames)

ds = xr.open_dataset(filenames[1])
#print(ds)
print(ds)
#T = ds.sel(lon=slice(-180, 180), lat=slice(-90, 90))
#T=T.get('umean')
#print(T.values)