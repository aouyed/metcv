import numpy as np
import xarray as xr
import pandas as pd
import second_stage.second_stage_run as ssr
from global_land_mask import globe
import datetime
SIGMA_LON = 1.5
SIGMA_LAT = 0.15
SIGMAS_0 = np.array([SIGMA_LON, SIGMA_LAT])


def main():
    pressure = 850
    filename = '../data/processed/experiments_12_17_20/3600_850_january.nc'
    triplet = datetime.datetime(2006, 1, 1, 0, 0)
    ds = xr.open_dataset(filename)
    ds = ds.loc[{'time': triplet, 'filter': 'df'}]
    print(ds)
    df = ssr.ds_to_dataframe(ds.copy(), triplet)
    df = df.drop(['filter', 'time'], axis=1)
    df = df.dropna()
    df = df.astype(np.float32)
    df_results = pd.DataFrame()
    with_fsua = False
    for factor in [0, 1, 5]:
        for with_fsua in [True, False]:
            sigmas = factor*SIGMAS_0
            print(sigmas)
            df_results_unit = ssr.loop(
                df, pressure, triplet, sigmas, with_fsua)
            df_results_unit['factor'] = factor
            df_results_unit['with_fsua'] = with_fsua
            if df_results.empty:
                df_results = df_results_unit
            else:
                df_results = pd.concat(
                    [df_results, df_results_unit]).reset_index()
    print(df_results)
    df_results.to_pickle('sensitivity.pkl')


if __name__ == '__main__':
    main()
