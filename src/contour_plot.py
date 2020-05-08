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


def gradient_plot(df, var):
    lat = df.pivot('lat', 'lon', 'lat').values
    lon = df.pivot('lat', 'lon', 'lon').values
    metv = df.pivot('lat', 'lon', var.lower()).values
    dx, dy = metpy.calc.lat_lon_grid_deltas(lon, lat)
    grad = metpy.calc.gradient(metv, deltas=(dy, dx))
    grad = grad.magnitude
    grad = np.nan_to_num(grad)
    return grad


def contourf_plotter(df,  values, title, units):

   # df['speed_error'] = np.sqrt(df['speed_error'])
    grid = 10
    var = df.pivot('lat', 'lon', values).values

    factor = 0.0625/grid

    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    # pmap = plt.cm.gnuplot
    pmap = plt.cm.coolwarm
    pmap.set_bad(color='grey')
    lon = np.arange(df['lon'].min(), df['lon'].max() + 0.0625, 0.0625)
    lat = np.arange(df['lat'].min(), df['lat'].max() + 0.0625, 0.0625)

    levs = [-30, -10, -5, -3, 0, 3, 5, 10, 30]
    im = ax.contourf(lon, lat, var, levs,  cmap=pmap,
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


dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


start_date = datetime.datetime(2006, 7, 1, 6, 0, 0, 0)
end_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)

df = aa.df_concatenator(dataframes_dict, start_date,
                        end_date, False, True, False)

grad = gradient_plot(df, 'qv')
print(grad)
