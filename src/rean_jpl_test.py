import xarray as xr
from second_stage import ml_functions as ml
import pandas as pd
import datetime
rmse = []
names = []

pressure = 850
filename = '../data/processed/experiments_12_17_20/3600_850_january_merged.nc'
triplet = datetime.datetime(2006, 1, 1, 0, 0)
ds = xr.open_dataset(filename)
print(ds)
ds_unit = ds.loc[{'time': ds['time'].values[0], 'filter': 'jpl'}].copy()
df = ds_unit.to_dataframe().reset_index(
    drop=True).dropna()
df = df.drop(['filter', 'time'], axis=1)
print(df)
_, _, _ = ml.error_calc(df.copy(), 'jpl', names, rmse)
_ = ml.error_rean(df, names, rmse)
d = {'rmse': rmse, 'filter': names}
df_results = pd.DataFrame(data=d)
print(df_results)
