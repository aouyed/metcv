import xarray as xr
import numpy as np
import xesmf as xe


ds = xr.open_dataset('MCS_Data4tracking_17_00.nc')
latmax = ds['lat'].max().item()
latmin = ds['lat'].min().item()
lonmax = ds['lon'].max().item()
lonmin = ds['lon'].min().item()

new_lat = np.arange(latmin, latmax, 0.0267)
new_lon = np.arange(lonmin, lonmax, 0.0267)

ds_out = xr.Dataset({'lat': (['lat'], new_lat), 'lon': ('lon', new_lon), })
regridder = xe.Regridder(ds, ds_out, 'bilinear')
dr_out = regridder(ds['qv_nr'])
breakpoint()
