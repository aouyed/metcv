import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import pdb
import glob


def big_histogram(dataframes, var, filter, column_x, column_y, s,  bins=100):
    """Creates a big histogram out of chunks in order to fit it in memory. """
    xedges = [np.inf, -np.inf]
    yedges = [np.inf, -np.inf]

    for df in dataframes:
        df = initialize_dataframe(filter, var, 5, df)
        xedges[0] = np.minimum(df[column_x].min(), xedges[0])
        xedges[1] = np.maximum(df[column_x].max(), xedges[1])

        yedges[0] = np.minimum(df[column_y].min(), yedges[0])
        yedges[1] = np.maximum(df[column_y].max(), yedges[1])

    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    heatmap = np.zeros((bins, bins), np.uint)
    for df in dataframes:
        df = initialize_dataframe(filter, var, 5, df)
        subtotal, _, _ = np.histogram2d(
            df[column_x], df[column_y], bins=[xbins, ybins])
        heatmap += subtotal.astype(np.uint)
    if s > 0:
        heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def histogram_plot(dataframes, var, filename, column_a, column_b, filter):
    """Initializes  histogram, plots it and saves it."""
    print('calculating histogram...')
    print(var)
    img, extent = big_histogram(
        dataframes, var,  filter, column_a, column_b, 1)
    print('plotting...')
    fig, ax = plt.subplots()
    im = ax.imshow(img, extent=extent, origin='lower',
                   cmap=cm.jet, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    plt.xlabel("wind speed [m/s]")
    plt.ylabel("speed difference [m/s] ")
    plt.tight_layout()
    plt.savefig('../histogram_'+filename+'.png', bbox_inches='tight', dpi=300)


def initialize_dataframe(filter, var, diff_limit, file):
    """Reads pickled dataframe and calculates important quantities such as wind speed."""
    print('initializing  ' + var + ' histogram...')
    df = pd.read_pickle(file)
    if filter is not 'jpl':
            df = df[df['filter'] == filter]
    df = df.drop_duplicates(['lat', 'lon'], keep='first')
    if var is 'angle':
        df = angle(df)
    df['speed'] = np.sqrt(df.umean**2+df.vmean**2)
    df['speed_track'] = np.sqrt(df.utrack**2+df.vtrack**2)
    df['speed_diff'] = df['speed_track']-df['speed']
    df = df[['speed_diff', var]].dropna()
    df = df[(abs(df['speed_diff']) <= diff_limit)]
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


dataframes = glob.glob('../../data/interim/experiments/dataframes/jpl/*')
#dataframes = glob.glob('../../data/interim/experiments/dataframes/ua/*')


#histogram_plot(dataframes, 'qv', 'ua_qv', 'qv', 'speed_diff', 'exp2')

# histogram_plot(dataframes, 'grad_mag_qv', 'ua_qv_mag',
#              'grad_mag_qv', 'speed_diff', 'exp2')

# histogram_plot(df, 'ua', 'speed', 'speed_diff')

# df = initialize_dataframe('exp2', 'angle', 5)
histogram_plot(dataframes, 'angle', 'jpl_angle', 'angle', 'speed_diff', 'jpl')
#histogram_plot(dataframes, 'angle', 'ua_angle', 'angle', 'speed_diff', 'exp2')

# df = initialize_dataframe('jpl', 'angle', 5)
# istogram_plot(df, 'jpl_angle', 'angle', 'speed_diff')

# df = initialize_dataframe('jpl', 'speed', 10)
# istogram_plot(df, 'jpl', 'speed', 'speed_diff')

# df = initialize_dataframe('df', 'speed', 10)
# istogram_plot(df, 'df', 'speed', 'speed_diff')
