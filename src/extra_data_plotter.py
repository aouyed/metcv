import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime
from viz import amv_analysis as aa
import pickle
import cartopy.crs as ccrs


def map_plotter(df,  values, title):

    df['speed_error'] = np.sqrt(df['speed_error'])
    grid = 10
    var = df.pivot('lat', 'lon', values).values

    factor = 0.0625/grid

    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    pmap = plt.cm.RdPu
    pmap.set_bad(color='grey')
    im = ax.imshow(var, cmap=pmap,
                   extent=[-180, 180, -90, 90], origin='lower')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    cbar.set_label('m/s ')
    plt.xlabel("lon")
    plt.ylabel("lat")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def line_plotter(df0, values, title):
    fig, ax = plt.subplots()

    df = df0[df0.categories == 'poly']
    ax.plot(np.array(df['latlon']), df['rmse'], '-o', label='poly')

    df = df0[df0.categories == 'rf']
    ax.plot(df['latlon'], df['rmse'], '-o', label='rf')

    df = df0[df0.categories == 'df']
    ax.plot(df['latlon'], df['rmse'], '-o', label='vem')

    df = df0[df0.categories == 'jpl']
    ax.plot(df['latlon'], df['rmse'], '-o', label='jpl')

    ax.legend(frameon=None)

    ax.set_xlabel("Region")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))

start_date = datetime.datetime(2006, 7, 1, 6, 0, 0, 0)
end_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)
dfc = aa.df_concatenator(dataframes_dict, start_date,
                         end_date, False, True, False)


dfc['qv'] = 1000*dfc['qv']
dfc = dfc.dropna(subset=['qv'])
dfc = dfc.dropna(subset=['umeanh'])
dfc = dfc[dfc['utrack'].isna()]
# dfc.to_pickle("df_0z.pkl")


map_plotter(dfc, 'qv', 'qv')

df0 = pd.read_pickle("./df_results.pkl")
df0.sort_values(by=['latlon'], inplace=True)

df = df0[(df0.extra == True) & (df0.test_size == 0.99)]
line_plotter(df, 'results_e', 'non-jpl, test_size=0.99')
df = df0[(df0.extra == False) & (df0.test_size == 0.99)]
line_plotter(df, 'results',  'jpl, test_size=0.99')


df = df0[(df0.extra == True) & (df0.test_size == 0.999)]
line_plotter(df, 'results_e_001', 'non-jpl, test_size=0.999')
df = df0[(df0.extra == False) & (df0.test_size == 0.999)]
line_plotter(df, 'results_001',  'jpl, test_size=0.999')
