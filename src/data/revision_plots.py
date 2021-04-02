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
    # xedges = [-10, 10]
    # yedges = [-10, 10]
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


def histogram_axes(ds, ds_1, colx, coly, colx1, coly1, extentx, extenty, title):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axlist = axes.flat
    hist, extent = vel_histogram(
        ds, colx, coly, extentx, extenty, bins=100)
    axlist[0] = histogram_plot(hist, extent, axlist[0])
    hist, extent = vel_histogram(
        ds_1, colx1, coly1, extentx, extenty,  bins=100)
    axlist[1] = histogram_plot(hist, extent, axlist[1])
    plt.savefig('../data/processed/plots/panel_histogram_'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def histogram_sequence(ds_train, ds_test):

    ds_rf = ds_test.sel(filter='exp2')
    ds_error = ds_test.sel(filter='ground_t')
    ds_df = ds_test.sel(filter='df')
    histogram_axes(ds_rf, ds_rf, 'umean', 'vmean', 'umean',
                   'vmean', [-10, 10], [-10, 10], 'vel')
    histogram_axes(ds_error, ds_error, 'utrack', 'umean',
                   'vtrack', 'vmean', [-5, 5], [-5, 5], 'fullvelnoise')
    histogram_axes(ds_df, ds_df, 'utrack', 'umean', 'vtrack',
                   'vmean', [-5, 5], [-5, 5], 'dfvelnoise')


def main(triplet, pressure=850, dt=3600):

    hist_dict = {}
    month = triplet.strftime("%B").lower()

    #   ds_name = str(dt)+'_' + str(pressure) + '_' + \
    #      triplet.strftime("%B").lower() + '_merged'
    ds_name = str(dt)+'_' + str(pressure) + '_train_' + \
        triplet.strftime("%B").lower()
    filename = PATH + ds_name+'.nc'
    ds_train = xr.open_dataset(filename)

    ds_name = str(dt)+'_' + str(pressure) + '_full_' + \
        triplet.strftime("%B").lower()
    filename = PATH + ds_name+'.nc'
    ds_full = xr.open_dataset(filename)

    ds_name = str(dt)+'_' + str(pressure) + '_' + \
        triplet.strftime("%B").lower() + '_merged'
    filename = PATH + ds_name+'.nc'
    ds_test = xr.open_dataset(filename)

    # ds_train = ds_train.sel(filter='exp2')
    histogram_sequence(ds_train, ds_test)


if __name__ == "__main__":
    main()
