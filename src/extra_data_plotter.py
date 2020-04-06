import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime
from viz import amv_analysis as aa
import pickle
import cartopy.crs as ccrs


def map_plotter(df,  values, title, units):

   # df['speed_error'] = np.sqrt(df['speed_error'])
    grid = 10
    var = df.pivot('lat', 'lon', values).values

    factor = 0.0625/grid

    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    pmap = plt.cm.gnuplot
    pmap.set_bad(color='grey')
    im = ax.imshow(var, cmap=pmap,
                   extent=[-180, 180, -90, 90], origin='lower')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    cbar.set_label(units)
    plt.xlabel("lon")
    plt.ylabel("lat")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def scatter_plotter(df,  values, title, units):
    grid = 10

    factor = 0.0625/grid

    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    pmap = plt.cm.RdPu

    im = ax.scatter(df['lon'], df['lat'], c=df[values],
                    s=1, cmap=pmap, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    plt.xlabel("lon")
    plt.ylabel("lat")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def freq_plotter(df,  values, title):

    fig, ax = plt.subplots()
    df = df[df[values] > 0]

    ax.plot(df[values], df['freq'], '-o', label='rf')
    plt.xlabel(values)
    plt.ylabel("freq")
    ax.set_title(title)
    plt.savefig(title+'.png', bbox_inches='tight', dpi=300)


def line_plotter(df0, values, title):
    fig, ax = plt.subplots()

    # df = df0[df0.categories == 'poly']
    # ax.plot(np.array(df['latlon']), df['rmse'], '-o', label='poly')

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
    plt.savefig('filtr'+'.png', bbox_inches='tight', dpi=300)


def filter_plotter(df0, values, title):
    fig, ax = plt.subplots()

    #df = df0[(df0.categories == 'rf') & (df0.exp_filter == False)]
    #ax.plot(df['latlon'], df['rmse'], '-o', label='rf, no filter')

    df = df0[(df0.categories == 'rf') & (df0.exp_filter == 'exp2')]
    ax.plot(df['latlon'], df['rmse'], '-o', label='rf, exponential noise')

    df = df0[(df0.categories == 'rf') & (df0.exp_filter == 'rand')]
    ax.plot(df['latlon'], df['rmse'], '-o',
            label='rf, no noise')

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


def main():
    dict_path = '../data/interim/dictionaries/dataframes.pkl'
    dataframes_dict = pickle.load(open(dict_path, 'rb'))

    df0 = pd.read_pickle("./df_results.pkl")
    df0.latlon[df0.latlon == '90°S,60°S'] = '(0) 90°S,60°S'
    df0.latlon[df0.latlon == '60°S,30°S'] = '(1) 60°S,30°S'
    df0.latlon[df0.latlon == '30°S,30°N'] = '(2) 30°S,30°N'
    df0.latlon[df0.latlon == '30°N,60°N'] = '(3) 30°N,60°N'
    df0.latlon[df0.latlon == '60°N,90°N'] = '(4) 60°N,90°N'
    print(df0)
    df0.sort_values(by=['latlon'], inplace=True)
    filter_plotter(df0, 'results_test', 'Exponential filter')


if __name__ == "__main__":
    main()
