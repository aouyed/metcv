import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import pdb
import glob
import cmocean
import matplotlib.colors as mcolors
from data import summary_statistics as ss
import xarray as xr
from joblib import Parallel, delayed
import pickle

PATH_DF = '../data/processed/dataframes/'

KG_TO_GRAMS = 1000
METERS_TO_KM = 1/1000
GRADIENT_TO_KM = KG_TO_GRAMS/METERS_TO_KM
HIST_X_EDGES = {'grad_mag_qv': [0, 0.05], 'qv': [
    0, 6], 'speed': [0, 30], 'angle': [-180, 180]}
CMAP = cm.CMRmap_r
VMAX_F = 1.5
PATH = '../data/processed/experiments/'


def vel_histogram(ds, column_x, column_y, xedges, yedges,  bins=100):
    """Creates a big histogram out of chunks in order to fit it in memory. """
    df = ds.to_dataframe().reset_index()
    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    heatmap = np.zeros((bins, bins), np.uint)
    print(column_y)
    print(column_x)
    print(df[[column_x, column_y]].corr())
    subtotal, _, _ = np.histogram2d(
        df[column_x], df[column_y], bins=[xbins, ybins], weights=df['cos_weight'])
    heatmap = subtotal.astype(np.uint)
    heatmap = heatmap/np.sum(heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    return heatmap.T, extent


def histogram_plot(hist, extent, ax):
    """Initializes  histogram, plots it and saves it."""
    print('calculating histogram...')
    im = ax.imshow(hist, extent=extent, origin='lower',
                   cmap=CMAP, aspect='auto')
    return ax


def histogram_axes(ds, ds_1, colx, coly, colx1, coly1, extentx, extenty):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axlist = axes.flat
    hist, extent = vel_histogram(
        ds, colx, coly, extentx, extenty, bins=100)
    axlist[0] = histogram_plot(hist, extent, axlist[0])
    hist, extent = vel_histogram(
        ds_1, colx1, coly1, extentx, extenty,  bins=100)
    axlist[1] = histogram_plot(hist, extent, axlist[1])
    return fig, axlist


def histogram_sequence(ds_train, ds_test):

    ds_rf = ds_test.sel(filter='exp2')
    ds_error = ds_test.sel(filter='ground_t')
    ds_df = ds_test.sel(filter='df')
    fig, axlist = histogram_axes(ds_train, ds_rf, 'umean', 'vmean', 'umean',
                                 'vmean', [-10, 10], [-10, 10])
    axlist[0].set(xlabel='u [m/s]')
    axlist[0].set(ylabel='v [m/s]')
    axlist[0].text(0.4, 0.85, 'train', transform=axlist[0].transAxes)
    axlist[1].text(0.4, 0.85, 'test', transform=axlist[1].transAxes)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/panel_histogram_vel.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    fig, axlist = histogram_axes(ds_error, ds_error, 'utrack', 'umean',
                                 'vtrack', 'vmean', [-5, 5], [-5, 5])
    axlist[0].set(ylabel='truth [m/s]')
    axlist[0].set(xlabel='noisy [m/s]')
    axlist[0].text(0.4, 0.85, 'u', transform=axlist[0].transAxes)
    axlist[1].text(0.4, 0.85, 'v', transform=axlist[1].transAxes)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/panel_histogram_velnoise.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    fig, axlist = histogram_axes(ds_df, ds_df, 'utrack', 'umean', 'vtrack',
                                 'vmean', [-5, 5], [-5, 5])
    axlist[0].set(ylabel='tracked [m/s]')
    axlist[0].set(xlabel='noisy [m/s]')
    axlist[0].text(0.4, 0.85, 'u', transform=axlist[0].transAxes)
    axlist[1].text(0.4, 0.85, 'v', transform=axlist[1].transAxes)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/panel_histogram_dfvelnoise.png',
                bbox_inches='tight', dpi=300)


def quiver(ax,  ds, vector_label):

    X, Y = np.meshgrid(ds['lon'].values, ds['lat'].values)

    Q = ax.quiver(X, Y, np.squeeze(
        ds[vector_label[0]].values), np.squeeze(ds[vector_label[1]].values), scale=40)

    qk = ax.quiverkey(
        Q, 0.8, 0.1, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='axes', labelcolor='red', color='r')

    # ax.set_aspect(1)
    return ax


def quiver_plot(ds, title):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(ds['lon'].values, ds['lat'].values)
    ax.set_title(title)
    Q = ax.quiver(X, Y, np.squeeze(
        ds['utrack'].values), np.squeeze(ds['vtrack'].values))
    qk = ax.quiverkey(Q, 0.8, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                      coordinates='figure')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_'+title+'.png',
                bbox_inches='tight', dpi=300)

    plt.close()


def histogram_initializer(triplet, pressure, dt):
    month = triplet.strftime("%B").lower()

    ds_name = str(dt)+'_' + str(pressure) + '_train_' + \
        triplet.strftime("%B").lower()
    filename = PATH + ds_name+'.nc'
    ds_train = xr.open_dataset(filename)

    ds_name = str(dt)+'_' + str(pressure) + '_' + \
        triplet.strftime("%B").lower() + '_merged'
    filename = PATH + ds_name+'.nc'
    ds_test = xr.open_dataset(filename)

    histogram_sequence(ds_train, ds_test)


def close_axes(axlist, title, fig):
    axlist[0].text(0.4, 0.85, 'ground truth', transform=axlist[0].transAxes)
    axlist[1].text(0.4, 0.85, 'JPL', transform=axlist[1].transAxes)
    axlist[2].text(0.4, 0.85, 'UA', transform=axlist[2].transAxes)
    axlist[3].text(0.4, 0.85, 'noisy', transform=axlist[3].transAxes)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_panel'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def error_computation(ds):
    ds['uerror'] = ds['utrack']-ds['umean']
    ds['verror'] = ds['vtrack']-ds['vmean']
    return ds


def quiver_axes(ds_gt, ds_jpl, ds_ua, ds_noise, title):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axlist = axes.flat
    axlist[0] = quiver(axlist[0], ds_gt, ('umean', 'vmean'))
    axlist[1] = quiver(axlist[1], ds_jpl, ('utrack', 'vtrack'))
    axlist[2] = quiver(axlist[2], ds_ua, ('utrack', 'vtrack'))
    axlist[3] = quiver(axlist[3], ds_noise, ('utrack', 'vtrack'))
    close_axes(axlist, 'vector ' + title, fig)

    ds_jpl = error_computation(ds_jpl)
    ds_ua = error_computation(ds_ua)
    ds_noise = error_computation(ds_noise)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axlist = axes.flat
    axlist[0] = quiver(axlist[0], ds_gt, ('umean', 'vmean'))
    axlist[1] = quiver(axlist[1], ds_jpl, ('uerror', 'verror'))
    axlist[2] = quiver(axlist[2], ds_ua, ('uerror', 'verror'))
    axlist[3] = quiver(axlist[3], ds_noise, ('uerror', 'verror'))
    close_axes(axlist, 'error ' + title, fig)


def quiver_plots(ds_df, ds_jpl, ds_ua):
    quiver_plot(ds_df, 'fsUA')
    quiver_plot(ds_jpl, 'JPL')
    quiver_plot(ds_ua, 'UA')


def quiver_initiation(ds_test, ds_full, minc, maxc, title, coarsen=False):
    ds_test = ds_test.loc[{'lon': slice(minc, maxc), 'lat': slice(minc, maxc)}]
    ds_full = ds_full.loc[{'lon': slice(minc, maxc), 'lat': slice(minc, maxc)}]
    if coarsen:
        ds_test = ds_test.coarsen(lon=10, boundary='trim').mean().coarsen(
            lat=10, boundary='trim').mean()
        ds_full = ds_full.coarsen(lon=10, boundary='trim').mean().coarsen(
            lat=10, boundary='trim').mean()

    quiver_axes(ds_test.loc[{'filter': 'df'}].copy(), ds_test.loc[{
        'filter': 'jpl'}].copy(), ds_full.loc[{'filter': 'full_exp2'}].copy(), ds_test.loc[{'filter': 'ground_t'}].copy(), title)


def main(triplet, pressure=850, dt=3600):
    ds_name = str(dt)+'_' + str(pressure) + '_' + \
        triplet.strftime("%B").lower() + '_merged'
    filename = PATH + ds_name+'.nc'
    ds_test = xr.open_dataset(filename)

    ds_name = str(dt)+'_' + str(pressure) + '_full_' + \
        triplet.strftime("%B").lower()
    filename = PATH + ds_name+'.nc'
    ds_full = xr.open_dataset(filename)
    print(ds_test)

    quiver_initiation(ds_test.copy(), ds_full.copy(), -
                      0.6, 0.6, 'micro', coarsen=False)
    quiver_initiation(ds_test.copy(), ds_full.copy(), -
                      6, 6, 'coarse', coarsen=True)
    histogram_initializer(triplet, pressure, dt)


if __name__ == "__main__":
    main()
