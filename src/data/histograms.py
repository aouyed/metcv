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


def big_histogram(ds, var, filter, column_x, column_y, prefix,  bins=100):
    """Creates a big histogram out of chunks in order to fit it in memory. """
    xedges = HIST_X_EDGES[column_x]
    yedges = [-7.5, 7.5]

    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    heatmap = np.zeros((bins, bins), np.uint)

    df = initialize_dataframe(filter, column_x, ds, prefix)
    subtotal, _, _ = np.histogram2d(
        df[column_x], df[column_y], bins=[xbins, ybins], weights=df['cos_weight'])
    heatmap += subtotal.astype(np.uint)
    heatmap = heatmap/np.sum(heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    return heatmap.T, extent


def histogram_dumper(img, extent, filename, hist_dict, column_a, filter):

    hist_dict[(filter, column_a, 'extent')] = extent
    hist_dict[(filter, column_a, 'img')] = img
    return hist_dict


def histogram_plot(ds, var, filename, column_a, column_b, filter, xlabel, hist_dict, prefix):
    """Initializes  histogram, plots it and saves it."""
    print('calculating histogram...')
    print(var)
    img, extent = big_histogram(
        ds, var,  filter, column_a, column_b, prefix)
    hist_dict = histogram_dumper(
        img, extent, filename, hist_dict, column_a, filter)
    print('plotting...')
    fig, ax = plt.subplots()
    if column_a in ('speed', 'angle', 'grad_mag_qv', 'qv'):
        if column_a == 'speed':
            vmax = 0.0015
        elif column_a == 'angle':
            vmax = 0.0008
        elif column_a == 'grad_mag_qv':
            vmax = 0.002
        else:
            vmax = 0.00175

        vmax = vmax*VMAX_F
        divnorm = mcolors.TwoSlopeNorm(vmin=0, vcenter=vmax/4, vmax=vmax)
        im = ax.imshow(img, extent=extent, origin='lower',
                       cmap=CMAP, aspect='auto', vmin=0, vmax=vmax, norm=divnorm)
    else:
        im = ax.imshow(img, extent=extent, origin='lower',
                       cmap=CMAP, aspect='auto')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    plt.xlabel(xlabel)
    plt.ylabel("speed difference [m/s] ")
    plt.tight_layout()
    plt.savefig('../data/processed/plots/histogram_' +
                filename+'.png', bbox_inches='tight', dpi=300)
    return hist_dict


def initialize_dataframe(filter_u, var,  ds, prefix):
    """Reads pickled dataframe and calculates important quantities such as wind speed."""

    if filter_u == 'reanalysis':
        ds = ds.sel(filter='df')
    else:
        ds = ds.sel(filter=filter_u)

    df = ds.to_dataframe()
    df = df.reset_index()
    if var is 'angle':
        df = angle(df)
    if var is 'qv':
        df[var] = df[var]*KG_TO_GRAMS
    if var is 'grad_mag_qv':
        df[var] = df[var]*GRADIENT_TO_KM
    df['speed'] = np.sqrt(df.umean**2+df.vmean**2)
    df['speed_track'] = np.sqrt(df.utrack**2+df.vtrack**2)
    df['speed_diff'] = df.speed_track - df.speed

    if filter is 'reanalysis':
        df['speed_diff'] = np.sqrt(df.u_error_rean**2+df.v_error_rean**2)
    df['cos_weight'] = np.cos(df['lat']/180*np.pi)
    df = df[['speed_diff', var, 'cos_weight']].dropna()
    df.to_pickle(PATH_DF+prefix+'_'+var+'_initialized.pkl')

    return df


def angle(df):
    """Calculates angle between moisture and wind velocity."""
    dot = df['grad_x_qv']*df['umean']+df['grad_y_qv']*df['vmean']
    mags = np.sqrt(df['grad_x_qv']**2+df['grad_y_qv']**2) * \
        np.sqrt(df['umean']**2+df['vmean']**2)
    c = (dot/mags)
    df['angle'] = np.arccos(c)
    df['angle'] = df.angle/np.pi*180
    df['neg_function'] = df['grad_x_qv'] * \
        df['vmean'] - df['grad_y_qv']*df['umean']
    df['angle'][df.neg_function < 0] = -df['angle'][df.neg_function < 0]
    df = df.drop(columns=['neg_function'])
    return df


def histogram_sequence(filter, prefix, ds, hist_dict):
    """Calculates batch of histogram plots"""
    hist_dict = histogram_plot(ds, 'speed', prefix + '_speed', 'speed',
                               'speed_diff', filter, 'Wind speed [m/s]', hist_dict, prefix)
    hist_dict = histogram_plot(ds, 'qv', prefix+'_qv', 'qv',
                               'speed_diff', filter, 'Moisture [g/kg]', hist_dict, prefix)
    hist_dict = histogram_plot(ds, 'grad_mag_qv', prefix+'_grad_mag_qv',
                               'grad_mag_qv', 'speed_diff', filter, 'Moisture gradient [g/(kg km)]', hist_dict, prefix)
    hist_dict = histogram_plot(ds, 'angle', prefix+'_angle', 'angle',
                               'speed_diff', filter, 'Wind-moisture gradient angle [deg]', hist_dict, prefix)
    return hist_dict


def main(triplet, pressure=500, dt=3600):

    hist_dict = {}
    month = triplet.strftime("%B").lower()

    ds_name = str(dt)+'_' + str(pressure) + '_' + \
        triplet.strftime("%B").lower() + '_merged'

    filename = PATH + ds_name+'.nc'
    ds = xr.open_dataset(filename)

    hist_dict = histogram_sequence('jpl', month+'_'+str(dt)+'_' +
                                   str(pressure) + '_jpl', ds, hist_dict)

    hist_dict = histogram_sequence('exp2', month+'_' + str(dt)+'_' +
                                   str(pressure)+'_ua', ds, hist_dict)
    hist_dict = histogram_sequence('df',  month+'_'+str(dt)+'_' +
                                   str(pressure)+'_df', ds, hist_dict)

    directory = '../data/processed/histograms/' + ds_name + '.pkl'
    f = open(directory, 'wb')
    pickle.dump(hist_dict, f)
    f.close()


if __name__ == "__main__":
    main()
