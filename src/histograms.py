import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import pdb


def myplot(x, y, s, bins=100):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    if s > 0:
        heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def histogram_plot(df, filename, column_a, column_b):
    print('calculating histogram...')
    img, extent = myplot(df[column_a], df[column_b], 1)
    print('plotting...')
    fig, ax = plt.subplots()
    im = ax.imshow(img, extent=extent, origin='lower',
                   cmap=cm.jet, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    plt.xlabel("wind speed [m/s]")
    plt.ylabel("speed difference [m/s] ")
    plt.tight_layout()
    plt.savefig('histogram_'+filename+'.png', bbox_inches='tight', dpi=300)


def initialize_dataframe(filter, var, diff_limit):
    print('initializing  ' + var + ' histogram...')
    df = pd.read_pickle('df_sample.pkl')
    df = df[df['filter'] == filter]
    if var is 'angle':
        df = angle(df)
    df['speed'] = np.sqrt(df.umean**2+df.vmean**2)
    df['speed_track'] = np.sqrt(df.utrack**2+df.vtrack**2)
    df['speed_diff'] = df['speed_track']-df['speed']
    df = df[['speed_diff', var]].dropna()
    df = df[(abs(df['speed_diff']) <= diff_limit)]
    return df


def angle_fun(grad_qv, vel_vector):
    dot = np.dot(grad_qv, vel_vector)
    mags = np.linalg.norm(grad_qv)*np.linalg.norm(vel_vector)
    c = (dot/mags)
    angle = np.arccos(c)
    angle = angle/np.pi*180
    neg_function = grad_qv[0]*vel_vector[1] - grad_qv[1]*vel_vector[0]
    if(neg_function < 0):
        angle = -angle
    return angle


def angle(df):
    df['vel_vector'] = list(zip(df.umean, df.vmean))
    df['grad_qv'] = list(zip(df.grad_x_qv, df.grad_y_qv))
    df.grad_qv = df['grad_qv'].apply(lambda x: np.array(x))
    df.vel_vector = df['vel_vector'].apply(lambda x: np.array(x))
    df['angle'] = df.apply(lambda x: angle_fun(
        x['grad_qv'], x['vel_vector']), axis=1)
    return df


df = initialize_dataframe('exp2', 'qv', 5)

histogram_plot(df, 'ua_qv', 'qv', 'speed_diff')


df = initialize_dataframe('exp2', 'grad_mag_qv', 5)
histogram_plot(df, 'ua_qv_mag', 'grad_mag_qv', 'speed_diff')

df = initialize_dataframe('exp2', 'speed', 5)
histogram_plot(df, 'ua', 'speed', 'speed_diff')

df = initialize_dataframe('exp2', 'angle', 5)
histogram_plot(df, 'ua_angle', 'angle', 'speed_diff')

df = initialize_dataframe('jpl', 'angle', 5)
histogram_plot(df, 'jpl_angle', 'angle', 'speed_diff')

df = initialize_dataframe('jpl', 'speed', 10)
histogram_plot(df, 'jpl', 'speed', 'speed_diff')

df = initialize_dataframe('df', 'speed', 10)
histogram_plot(df, 'df', 'speed', 'speed_diff')
