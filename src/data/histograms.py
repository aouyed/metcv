import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import pdb
import glob
import cmocean
import matplotlib.colors as mcolors

KG_TO_GRAMS = 1000
METERS_TO_KM = 1/1000
GRADIENT_TO_KM = KG_TO_GRAMS/METERS_TO_KM
HIST_X_EDGES = {'grad_mag_qv': [0, 0.05], 'qv': [
    0, 6], 'speed': [0, 30], 'angle': [-180, 180]}


def big_histogram(dataframes, var, filter, column_x, column_y, s,  bins=100):
    """Creates a big histogram out of chunks in order to fit it in memory. """
    xedges = HIST_X_EDGES[column_x]
    yedges = [-7.5, 7.5]

    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    heatmap = np.zeros((bins, bins), np.uint)
    for df in dataframes:
        print(df)
        df = initialize_dataframe(filter, var, df)
        subtotal, _, _ = np.histogram2d(
            df[column_x], df[column_y], bins=[xbins, ybins])
        heatmap += subtotal.astype(np.uint)
    if s > 0:
        heatmap = gaussian_filter(heatmap, sigma=s)
        heatmap = heatmap/np.sum(heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def histogram_plot(dataframes, var, filename, column_a, column_b, filter, xlabel):
    """Initializes  histogram, plots it and saves it."""
    print('calculating histogram...')
    print(var)
    img, extent = big_histogram(
        dataframes, var,  filter, column_a, column_b, 1)
    print('plotting...')
    fig, ax = plt.subplots()
    if column_a in ('speed', 'angle'):
        if column_a is 'speed':
            vmax = 0.0015
        else:
            vmax = 0.0008
        divnorm = mcolors.TwoSlopeNorm(vmin=0, vcenter=vmax/4, vmax=vmax)
        im = ax.imshow(img, extent=extent, origin='lower',
                       cmap=cm.jet, aspect='auto', vmin=0, vmax=vmax, norm=divnorm)
    else:
        im = ax.imshow(img, extent=extent, origin='lower',
                       cmap=cm.jet, aspect='auto')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    plt.xlabel(xlabel)
    plt.ylabel("speed difference [m/s] ")
    plt.tight_layout()
    plt.savefig('../data/processed/plots/histogram_' +
                filename+'.png', bbox_inches='tight', dpi=300)


def initialize_dataframe(filter, var,  file):
    """Reads pickled dataframe and calculates important quantities such as wind speed."""
    print('initializing  ' + var + ' histogram...')
    df = pd.read_pickle(file)
    if filter is not 'jpl':
        if filter is 'reanalysis':
            df = df[df['filter'] == 'df']
        else:
            df = df[df['filter'] == filter]
    df = df.drop_duplicates(['lat', 'lon'], keep='first')
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
    df = df[['speed_diff', var]].dropna()
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


def histogram_sequence(filter, prefix, dataframes):
    """Calculates batch of histogram plots"""
    histogram_plot(dataframes, 'speed', prefix + '_speed', 'speed',
                   'speed_diff', filter, 'Wind speed [m/s]')
    histogram_plot(dataframes, 'qv', prefix+'_qv', 'qv',
                   'speed_diff', filter, 'Moisture [g/kg]')
    histogram_plot(dataframes, 'grad_mag_qv', prefix+'_grad_mag_qv',
                   'grad_mag_qv', 'speed_diff', filter, 'Moisture gradient [g/(kg km)]')
    histogram_plot(dataframes, 'angle', prefix+'_angle', 'angle',
                   'speed_diff', filter, 'Wind-moisture gradient angle [deg]')


def main(triplet, pressure=500, dt=3600):
    dataframes = glob.glob('../data/interim/experiments/dataframes/ua/*')
    month = triplet.strftime("%B").lower()

    histogram_sequence('exp2', month+'_' + str(dt)+'_' +
                       str(pressure)+'_ua', dataframes)
    histogram_sequence('df',  month+'_'+str(dt)+'_' +
                       str(pressure)+'_df', dataframes)
    histogram_sequence('reanalysis', month+'_' + str(dt)+'_' +
                       str(pressure)+'_rean', dataframes)
    histogram_sequence('ground_t', str(dt)+'_'+str(pressure)+'_gt', dataframes)

    dataframes = glob.glob('../data/interim/experiments/dataframes/jpl/*')
    histogram_sequence('jpl', month+'_'+str(dt)+'_' +
                       str(pressure) + '_jpl', dataframes)


if __name__ == "__main__":
    main()
