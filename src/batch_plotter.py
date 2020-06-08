import pdb
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime

file = '../data/processed/experiments/july.nc'
date = datetime(2006, 7, 1, 0, 0, 0, 0)
ds = xr.open_dataset(file)
ds_track = xr.open_dataset(
    '../data/interim/experiments/july/tracked/60min/1.nc')
ds = ds.sel(time=date)
ds_track = ds_track.sel(time=date)
ds_track = ds_track.expand_dims('filter')
ds_track = ds_track.assign_coords(filter=['jpl'])


df = ds.to_dataframe().reset_index()
dft = ds_track.to_dataframe().reset_index()
df_tot = df.merge(dft.dropna(), on=[
                  'lat', 'lon'], how='left', indicator='Exist')
df_tot = df_tot[df_tot.utrack_y.notna()]


pdb.set_trace()

fig, ax = plt.subplots()

df = df0[(df0.categories == 'rf') & (df0.exp_filter == 'exp2')]
ax.plot(df['latlon'], df['rmse'], '-o', label='UA (RF+VEM)')

df = df0[(df0.categories == 'ground_t') & (df0.exp_filter == 'ground_t')]
ax.plot(df['latlon'], df['rmse'], '-o',
        label='noisy observations')

df = df0[df0.categories == 'df']
ax.plot(df['latlon'], df['rmse'], '-o', label='VEM')

df = df0[df0.categories == 'jpl']
ax.plot(df['latlon'], df['rmse'], '-o', label='JPL')

ax.legend(frameon=None)
ax.set_ylim(0, 5)
ax.set_xlabel("Region")
ax.set_ylabel("RMSVD [m/s]")
ax.set_title(title)
directory = '../data/processed/density_plots'
plt.savefig(values+'.png', bbox_inches='tight', dpi=300)
