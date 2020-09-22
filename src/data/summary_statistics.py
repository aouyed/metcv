import pandas as pd
from data import histograms as hist
import numpy as np
import glob
import xarray as xr
import time
PATH_DF = '../data/processed/dataframes/'
PATH = '../data/processed/experiments/'


def summary_stats(df_stats, skew_dict, filter, column_y, var):
    print('starting summary statistics...')
    mean = df_stats['weighed'].sum()/df_stats['cos_weight'].sum()
    df_stats['weighed'] = df_stats['cos_weight']*(df_stats[column_y]-mean)**3
    k3 = df_stats['weighed'].sum()/df_stats['cos_weight'].sum()
    df_stats['weighed'] = df_stats['cos_weight']*(df_stats[column_y]-mean)**2
    k2 = df_stats['weighed'].sum()/df_stats['cos_weight'].sum()
    skewness = k3/k2**(3/2)
    stdev = np.sqrt(k2)

    skew_dict['filter'].append(filter)
    skew_dict['var'].append(var)
    skew_dict['mean'].append(mean)
    skew_dict['skewness'].append(skewness)
    skew_dict['stdev'].append(stdev)
    skew_dict['q50'].append(abs(df_stats[column_y]).quantile(0.5))
    skew_dict['q68'].append(abs(df_stats[column_y]).quantile(0.68))
    skew_dict['q95'].append(abs(df_stats[column_y]).quantile(0.95))
    print(skew_dict)

    return skew_dict


def stat_calculator(filter, column_y, column_x, ds, skew_dict):
    start_time = time.time()
    print('initializing dataframe')
    df = hist.initialize_dataframe(filter, column_x, ds)
    print("--- seconds ---" + str(time.time() - start_time))
    if column_x == 'angle':
        print('size of complete df: ' + str(df.shape))
        df_unit = df[(abs(df.angle) <= 91) & (abs(df.angle) >= 89)]
        df_unit2 = df[(abs(df.angle) <= 181) & (abs(df.angle) >= 179)]
        df = pd.concat([df_unit, df_unit2])
        print('size for angle=90: ' + str(df.shape))

    df_stats = df[['cos_weight', column_y]]
    df_stats['weighed'] = df_stats[column_y]*df_stats['cos_weight']
    skew_dict = summary_stats(df_stats, skew_dict, filter, column_y, column_x)
    return skew_dict


def main(triplet, pressure=500, dt=3600):

    skew_dict = {'filter': [], 'var': [],  'mean': [],
                 'skewness': [], 'stdev': [], 'q50': [], 'q68': [], 'q95': []}

    month = triplet.strftime("%B").lower()

    ds_name = str(dt)+'_' + str(pressure) + '_' + \
        triplet.strftime("%B").lower() + '_merged'

    filename = PATH + ds_name+'.nc'
    ds = xr.open_dataset(filename)

    skew_dict = stat_calculator(
        'jpl', 'speed_diff', 'angle', ds, skew_dict)
    skew_dict = stat_calculator(
        'jpl', 'speed_diff', 'speed', ds, skew_dict)
    skew_dict = stat_calculator(
        'exp2', 'speed_diff', 'speed', ds, skew_dict)
    skew_dict = stat_calculator(
        'df', 'speed_diff', 'speed', ds, skew_dict)
    skew_dict = stat_calculator(
        'exp2', 'speed_diff', 'angle', ds, skew_dict)
    skew_dict = stat_calculator(
        'df', 'speed_diff', 'angle', ds, skew_dict)

    df = pd.DataFrame(skew_dict)
    print(df)
    df.to_pickle(PATH_DF+str(dt)+'_'+month+'_'+str(pressure)+'_df_stats.pkl')


if __name__ == "__main__":
    main()
