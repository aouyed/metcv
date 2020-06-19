import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime
from viz import amv_analysis as aa
import pickle
import cartopy.crs as ccrs
import cv2
import matplotlib.colors as mcolors
import metpy as metpy


def quiver_plotter(df, title, uname, vname):
    grid = 10
    U = df.pivot('lat', 'lon', uname).values
    V = df.pivot('lat', 'lon', vname).values

    factor = 0.0625/grid

    U = cv2.resize(U, None, fx=factor, fy=factor)
    V = cv2.resize(V, None, fx=factor, fy=factor)
    X = np.arange(-180, 180 - grid, grid)
    Y = np.arange(-90, 90 - grid, grid)
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    gridlines = ax.gridlines(alpha=1, draw_labels=True)
    gridlines.xlabels_top = False

    gridlines.xlines = False
    gridlines.ylines = False
    Q = ax.quiver(X, Y, U, V, pivot='middle',
                  transform=ccrs.PlateCarree(), scale=250)
    qk = ax.quiverkey(Q, 0.9, 0.825, 10, r'$10 \frac{m}{s}$', labelpos='E',
                      coordinates='figure')

    ax.set_title('Observed Velocities')
    directory = '../data/processed/density_plots'
    plt.savefig(title+'.png', bbox_inches='tight', dpi=300)
    print('plotted quiver...')


def map_plotter(df,  values, title, units, vmin, vmax):

   # df['speed_error'] = np.sqrt(df['speed_error'])
    grid = 10
    var = df.pivot('lat', 'lon', values).values

    factor = 0.0625/grid

    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    pmap = plt.cm.gnuplot
    # pmap = plt.cm.coolwarm
    pmap.set_bad(color='grey')
    if abs(vmax) > 0:
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vmax/4, vmax=vmax)
        im = ax.imshow(var, cmap=pmap,
                       extent=[-180, 180, -90, 90], origin='lower', vmin=vmin, vmax=vmax, norm=divnorm)
    else:
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
    plt.savefig(title+'.png', bbox_inches='tight', dpi=300)


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


def filter_plotter(df0, values, title):
    fig, ax = plt.subplots()

    df = df0[df0.exp_filter == 'exp2']
    ax.plot(df['latlon'], df['rmse'], '-o', label='UA (RF+VEM)')

    df = df0[df0.exp_filter == 'ground_t']
    ax.plot(df['latlon'], df['rmse'], '-o',
            label='error from reanalysis')

    df = df0[df0.exp_filter == 'df']
    ax.plot(df['latlon'], df['rmse'], '-o', label='VEM')

    df = df0[df0.exp_filter == 'jpl']
    ax.plot(df['latlon'], df['rmse'], '-o', label='JPL')

    ax.legend(frameon=None)
    ax.set_ylim(0, 8)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def sorting_latlon(df0):
    df0.latlon[df0.latlon == '90°S,60°S'] = '(0) 90°S,60°S'
    df0.latlon[df0.latlon == '60°S,30°S'] = '(1) 60°S,30°S'
    df0.latlon[df0.latlon == '30°S,30°N'] = '(2) 30°S,30°N'
    df0.latlon[df0.latlon == '30°N,60°N'] = '(3) 30°N,60°N'
    df0.latlon[df0.latlon == '60°N,90°N'] = '(4) 60°N,90°N'
    print(df0)
    df0.sort_values(by=['latlon'], inplace=True)
    return df0


def main():
    dict_path = '../data/interim/dictionaries/dataframes.pkl'
    dataframes_dict = pickle.load(open(dict_path, 'rb'))

    #  filter_plotter(df0, 'results_test', 'training data size = 5%')


if __name__ == "__main__":
    main()
