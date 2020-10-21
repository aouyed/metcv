import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import pdb
import glob
import cmocean
import matplotlib.colors as mcolors
import xarray as xr
from joblib import Parallel, delayed
import pickle

KG_TO_GRAMS = 1000
METERS_TO_KM = 1/1000
GRADIENT_TO_KM = KG_TO_GRAMS/METERS_TO_KM
HIST_X_EDGES = {'grad_mag_qv': [0, 0.05], 'qv': [
    0, 6], 'speed': [0, 30], 'angle': [-180, 180]}
CMAP = cm.CMRmap_r
VMAX_F = 1.5
PATH = '../data/processed/experiments/'


def histogram_plot(column_a,  hist_dict, axlist, axpos):
    """Initializes  histogram, plots it and saves it."""
    print('calculating histogram...')
    print(column_a)
    vmax = 0.003*100
    for filter in ('exp2', 'df', 'jpl'):
        ax = axlist[axpos]
        img = hist_dict[(filter, column_a, 'img')]
        extent = hist_dict[(filter, column_a, 'extent')]
        divnorm = mcolors.TwoSlopeNorm(vmin=0, vcenter=vmax/4, vmax=vmax)
        im = ax.imshow(100*img, extent=extent, origin='lower',
                       cmap=CMAP, aspect='auto', vmin=0, vmax=vmax, norm=divnorm)
        axpos = axpos+1
    return im, axpos, axlist


def histogram_sequence(hist_dict, axlist):
    """Calculates batch of histogram plots"""
    axpos = 0
    _, axpos, axlist = histogram_plot('speed', hist_dict, axlist, axpos)
    _, axpos, axlist = histogram_plot('grad_mag_qv',  hist_dict, axlist, axpos)
    im, axpos, axlist = histogram_plot('angle', hist_dict, axlist, axpos)
    return im


def main(triplet, pressure=500, dt=3600):

    month = triplet.strftime("%B").lower()
    ds_name = str(dt)+'_' + str(pressure) + '_' + \
        triplet.strftime("%B").lower() + '_merged'
    directory = '../data/processed/histograms/' + ds_name + '.pkl'
    f = open(directory, 'rb')
    hist_dict = pickle.load(f)
    f.close()

    fig, axes = plt.subplots(nrows=3, ncols=3)
    axlist = axes.flat
    im = histogram_sequence(hist_dict, axlist)
    cbar_ax = fig.add_axes([0.12, -0.07, 0.77, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='percent')
    axlist[0].text(0.4, 0.85, 'UA', transform=axlist[0].transAxes)
    axlist[1].text(0.4, 0.85, 'fsUA', transform=axlist[1].transAxes)
    axlist[2].text(0.4, 0.85, 'JPL', transform=axlist[2].transAxes)
    axlist[3].set(ylabel='Speed difference [m/s]')
    axlist[1].set(xlabel='Wind speed [m/s]')
    axlist[4].set(xlabel='Moisture gradient [g/(kg km)]')
    axlist[7].set(xlabel='Wind-moisture gradient angle [deg]')

    # fig.tight_layout()
    fig.subplots_adjust(hspace=0.7)
    plt.savefig('../data/processed/plots/panel_histogram_' +
                ds_name+'.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
