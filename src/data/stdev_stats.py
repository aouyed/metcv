import xarray as xr
import weightedcalcs as wc
import numpy as np
import pandas as pd

delta = 30


def main():
    ds = xr.open_dataset(
        '../data/processed/experiments/3600_850_january_merged.nc')

    metric = {'region': [], 'variable': [],
              'UA (error)': [], 'truth (std)': []}

    ds = ds.loc[{'lat': slice(-30, 30), 'filter': 'exp2',
                 'time': str(ds.time.values[0])}].copy()
    ds['speed'] = np.sqrt(ds['umean']**2 + ds['vmean']**2)
    ds['speed_track'] = np.sqrt(ds['utrack']**2 + ds['vtrack']**2)
    ds['speed_diff'] = ds['speed_track']-ds['speed']
    ds['u_error'] = ds['utrack']-ds['umean']
    ds['v_error'] = ds['vtrack']-ds['vmean']
    ds['mag_squared'] = ds['u_error']**2+ds['v_error']**2

    for center in (0, -90, 120):
        print('region ' + str(center))
        ds_unit = ds.loc[{'lon': slice(center - delta, center + delta)}].copy()
        print(ds_unit)
        df = ds_unit[['mag_squared', 'u_error', 'v_error', 'utrack', 'vtrack', 'umean', 'vmean', 'speed', 'speed_diff',
                      'cos_weight']].to_dataframe().reset_index().dropna()
        df = df.rename(columns={'umean': 'u', 'vmean': 'v'})
        for variables in (('speed', 'speed_diff'), ('u', 'u_error'), ('v', 'v_error'), ('rmsvd', 'std mag')):
            metric['region'].append(center)
            metric['variable'].append(variables[0])
            calc = wc.Calculator("cos_weight")

            if variables[0] != 'rmsvd':
                stdev_tr = calc.std(df, variables[0])
                mean_ua = calc.mean(df, variables[1])
            else:
                u_std = calc.std(df, 'u')
                v_std = calc.std(df, 'v')
                stdev_tr = np.sqrt(u_std**2+v_std**2)
                mean_ua = np.sqrt(calc.mean(df, 'mag_squared'))

            metric['UA (error)'].append(mean_ua)
            metric['truth (std)'].append(stdev_tr)

    print(metric)
    df = pd.DataFrame(metric)

    print(df)
    print(df.round(3).to_latex(index=False))


if __name__ == "__main__":
    main()
