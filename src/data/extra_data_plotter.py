import metpy as metpy
import matplotlib.colors as mcolors
import cv2
import cartopy.crs as ccrs
import pickle
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cmocean

ERROR_MAX = 10.5


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


def map_plotter(var, title, units, vmin, vmax):
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    #pmap = plt.cm.gnuplot
    pmap = cmocean.cm.haline
    #pmap = plt.cm.winter
    # pmap.set_bad(color='grey')
    if abs(vmax) > 0:
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vmax/4, vmax=vmax)
        im = ax.imshow(var, cmap=pmap,
                       extent=[-180, 180, -90, 90], origin='lower', vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(
            var, cmap=pmap, extent=[-180, 180, -90, 90], origin='lower')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    cbar.set_label(units)
    plt.xlabel("lon")
    plt.ylabel("lat")
    # ax.set_title(title)
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
    plt.close()


def map_plotter_multiple(ds, ds_full, ds_jpl, title, units, vmin, vmax):
    fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                             'projection': ccrs.PlateCarree()})

    pmap = cmocean.cm.haline
    axlist = axes.flat
    axlist[0].coastlines()

    var = ds['error_mag'].loc[dict(filter='df')].values
    var = np.squeeze(var)
    im = axlist[0].imshow(var, cmap=pmap,
                          extent=[-180, 180, -90, 90], origin='lower', vmin=vmin, vmax=vmax)
    axlist[0].set_title('(a) fsUA')

    axlist[1].coastlines()
    var = ds_jpl['error_mag'].values
    var = np.squeeze(var)
    im = axlist[1].imshow(var, cmap=pmap,
                          extent=[-180, 180, -90, 90], origin='lower', vmin=vmin, vmax=vmax)
    axlist[1].set_title('(b) JPL')

    axlist[2].coastlines()
    var = ds_full['error_mag'].loc[dict(filter='full_exp2')].values
    var = np.squeeze(var)
    im = axlist[2].imshow(var, cmap=pmap,
                          extent=[-180, 180, -90, 90], origin='lower', vmin=vmin, vmax=vmax)
    axlist[2].set_title('(c) UA')

    axlist[3].coastlines()
    var = ds['error_mag'].loc[dict(filter='ground_t')].values
    var = np.squeeze(var)
    im = axlist[3].imshow(var, cmap=pmap,
                          extent=[-180, 180, -90, 90], origin='lower', vmin=vmin, vmax=vmax)
    axlist[3].set_title('(d) Model Error')

    cbar_ax = fig.add_axes([0.12, 0.125, 0.78, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='[m/s]')
    fig.subplots_adjust(hspace=-0.3, wspace=0.05)
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
    plt.close()


def map_plotter_multiple_better(ds, ds_full, ds_jpl, title, units, vmin, vmax):
    fig, axes = plt.subplots(nrows=2, ncols=2)

    pmap = cmocean.cm.haline
    axlist = axes.flat

    var = ds['error_mag'].loc[dict(filter='df')].values
    var = np.squeeze(var)

    extent_values = [ds_full['lon'].min().item(), ds_full['lon'].max(
    ).item(), ds_full['lat'].min().item(), ds_full['lat'].max().item()]

    im = axlist[0].imshow(var, cmap=pmap,
                          extent=extent_values, origin='lower', vmin=vmin, vmax=vmax)
    axlist[0].set_title('(a) fsUA')
    axlist[0].xaxis.set_tick_params(labelsize='small')
    axlist[0].yaxis.set_tick_params(labelsize='small')
    var = ds_jpl['error_mag'].values
    var = np.squeeze(var)
    im = axlist[1].imshow(var, cmap=pmap,
                          extent=extent_values, origin='lower', vmin=vmin, vmax=vmax)
    axlist[1].set_title('(b) JPL')
    axlist[1].xaxis.set_tick_params(labelsize='small')
    axlist[1].yaxis.set_tick_params(labelsize='small')

    var = ds_full['error_mag'].loc[dict(filter='full_exp2')].values
    var = np.squeeze(var)
    im = axlist[2].imshow(var, cmap=pmap,
                          extent=extent_values, origin='lower', vmin=vmin, vmax=vmax)
    axlist[2].set_title('(c) UA')
    axlist[2].xaxis.set_tick_params(labelsize='small')
    axlist[2].yaxis.set_tick_params(labelsize='small')

    var = ds['error_mag'].loc[dict(filter='ground_t')].values
    var = np.squeeze(var)
    im = axlist[3].imshow(var, cmap=pmap,
                          extent=extent_values, origin='lower', vmin=vmin, vmax=vmax)
    axlist[3].set_title('(d) Model Error')
    axlist[3].xaxis.set_tick_params(labelsize='small')
    axlist[3].yaxis.set_tick_params(labelsize='small')

    cbar_ax = fig.add_axes([0.12, 0.125, 0.78, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='[m/s]')
    fig.subplots_adjust(hspace=-0.3, wspace=0.16)
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
    plt.close()


def results_plotter(ax, df0):

    df0.latlon[df0.latlon == '(0) 90°S,60°S'] = '90°S,60°S'
    df0.latlon[df0.latlon == '(1) 60°S,30°S'] = '60°S,30°S'
    df0.latlon[df0.latlon == '(2) 30°S,30°N'] = '30°S,30°N'
    df0.latlon[df0.latlon == '(3) 30°N,60°N'] = '30°N,60°N'
    df0.latlon[df0.latlon == '(4) 60°N,90°N'] = '60°N,90°N'

    df = df0[df0.exp_filter == 'exp2']

    ax.plot(df['latlon'], df['rmse'], '-o', label='UA')

    df = df0[df0.exp_filter == 'ground_t']
    ax.plot(df['latlon'], df['rmse'], '-o',
            label='Model Error')

  #  df = df0[df0.exp_filter == 'rean']
   # ax.plot(df['latlon'], df['rmse'], '-o',
    #        label='Reanalysis Error')

    df = df0[df0.exp_filter == 'df']
    ax.plot(df['latlon'], df['rmse'], '-o', label='fsUA')

    df = df0[df0.exp_filter == 'jpl']
    ax.plot(df['latlon'], df['rmse'], '-o', label='JPL')
    ax.set_ylim(1, ERROR_MAX)
    ax.tick_params(axis='x', which='major', labelsize=6.5)


def multiple_filter_plotter(df_dict, values, month):

    fig, ax = plt.subplots()

    fig, axes = plt.subplots(nrows=2, ncols=2)

    axlist = axes.flat

    df = df_dict[(3600, month, 850)]
    results_plotter(axlist[0], df)
    axlist[0].set_ylabel("RMSVD [m/s]")

    df = df_dict[(3600, month, 500)]
    results_plotter(axlist[1], df)
    df = df_dict[(1800, month, 850)]
    results_plotter(axlist[2], df)
    axlist[2].set_ylabel("RMSVD [m/s]")
    df = df_dict[(1800, month, 500)]
    results_plotter(axlist[3], df)

    axlist[0].text(0.25, 0.7, '(a)\nΔt = 60 min\nP = 850 hPa',
                   transform=axlist[0].transAxes)
    axlist[1].text(0.25, 0.7, '(b)\nΔt = 60 min\nP = 500 hPa',
                   transform=axlist[1].transAxes)
    axlist[2].text(0.25, 0.7, '(c)\nΔt = 30 min\nP = 850 hPa',
                   transform=axlist[2].transAxes)
    axlist[3].text(0.25, 0.7, '(d)\nΔt = 30 min\nP = 500 hPa',
                   transform=axlist[3].transAxes)

    handles, labels = axlist[3].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(
        0.715, 1), ncol=2, frameon=False)
    #fig.legend(handles, labels, loc=(0.5, 1), ncol=2)
    plt.savefig(values+'.png', bbox_inches='tight',  dpi=300)
    plt.close()


def sorting_latlon(df0):
    df0.latlon[df0.latlon == '90°S,60°S'] = '(0) 90°S,60°S'
    df0.latlon[df0.latlon == '60°S,30°S'] = '(1) 60°S,30°S'
    df0.latlon[df0.latlon == '30°S,30°N'] = '(2) 30°S,30°N'
    df0.latlon[df0.latlon == '30°N,60°N'] = '(3) 30°N,60°N'
    df0.latlon[df0.latlon == '60°N,90°N'] = '(4) 60°N,90°N'
    print(df0)
    df0.sort_values(by=['latlon'], inplace=True)
    return df0


def filter_plotter(df0, values, title):
    fig, ax = plt.subplots()

    df = df0[df0.exp_filter == 'exp2']
    ax.plot(df['latlon'], df['rmse'], '-o', label='UA')

    df = df0[df0.exp_filter == 'ground_t']
    ax.plot(df['latlon'], df['rmse'], '-o',
            label='model error')

    df = df0[df0.exp_filter == 'df']
    ax.plot(df['latlon'], df['rmse'], '-o', label='UA First Stage')

    df = df0[df0.exp_filter == 'jpl']
    ax.plot(df['latlon'], df['rmse'], '-o', label='JPL')

    ax.legend(frameon=None)
    ax.set_ylim(0, ERROR_MAX)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    dict_path = '../data/interim/dictionaries/dataframes.pkl'
    dataframes_dict = pickle.load(open(dict_path, 'rb'))

    #  filter_plotter(df0, 'results_test', 'training data size = 5%')


if __name__ == "__main__":
    main()
