import pandas as pd
from data import histograms as hist
import numpy as np
import glob


def stat_calculator(filter, column_y, dataframes, skew_dict):

    df_stats = pd.DataFrame()

    for df in dataframes:
        #df = pd.read_pickle(df)
        print(df)
        df = hist.initialize_dataframe(filter, 'speed', df)
        if not df_stats.empty:
            df_stats = pd.concat([df_stats, df[['cos_weight', column_y]]])
        else:
            df_stats = df[['cos_weight', column_y]]

    df_stats['weighed'] = df_stats[column_y]*df_stats['cos_weight']
    mean = df_stats['weighed'].sum()/df_stats['cos_weight'].sum()
    df_stats['weighed'] = df_stats['cos_weight']*(df_stats[column_y]-mean)**3
    k3 = df_stats['weighed'].sum()/df_stats['cos_weight'].sum()
    df_stats['weighed'] = df_stats['cos_weight']*(df_stats[column_y]-mean)**2
    k2 = df_stats['weighed'].sum()/df_stats['cos_weight'].sum()
    skewness = k3/k2**(3/2)
    stdev = np.sqrt(k2)

    skew_dict['filter'].append(filter)
    skew_dict['mean'].append(mean)
    skew_dict['skewness'].append(skewness)
    skew_dict['stdev'].append(stdev)
    return skew_dict


def main(triplet, pressure=500, dt=3600):

    skew_dict = {'filter': [], 'skewness': [], 'stdev': [], 'mean': []}

    month = triplet.strftime("%B").lower()

    dataframes = glob.glob('../data/interim/experiments/dataframes/jpl/*')
    stat_calculator('jpl', 'speed_diff', dataframes, skew_dict)

    dataframes = glob.glob('../data/interim/experiments/dataframes/ua/*')
    stat_calculator('exp2', 'speed_diff', dataframes, skew_dict)
    stat_calculator('reanalysis', 'speed_diff', dataframes, skew_dict)
    stat_calculator('df', 'speed_diff', dataframes, skew_dict)

    df = pd.DataFrame(skew_dict)
    print(df)
    df.to_pickle(str(dt)+'_'+month+'_'+str(pressure)+'_df_stats.pkl')


if __name__ == "__main__":
    main()
