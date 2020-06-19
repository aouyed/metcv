import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import pandas as pd


def myplot(x, y, s, bins=100):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    print(heatmap.shape)
    print(heatmap)
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


df = pd.read_pickle('df_sample.pkl')
df = df[df['filter'] == 'exp2']
df['speed'] = np.sqrt(df.umean**2+df.vmean**2)
df['speed_track'] = np.sqrt(df.utrack**2+df.vtrack**2)
df['speed_diff'] = df['speed_track']-df['speed']
df = df[['speed_diff', 'qv']].dropna()
df = df[(abs(df['speed_diff']) <= 5)]

histogram_plot(df, 'ua_qv', 'qv', 'speed_diff')
df = pd.read_pickle('df_sample.pkl')
df = df[df['filter'] == 'exp2']
df['speed'] = np.sqrt(df.umean**2+df.vmean**2)
df['speed_track'] = np.sqrt(df.utrack**2+df.vtrack**2)
df['speed_diff'] = df['speed_track']-df['speed']
df = df[['speed_diff', 'grad_mag_qv']].dropna()
df = df[(abs(df['speed_diff']) <= 5)]
histogram_plot(df, 'ua_qv_mag', 'grad_mag_qv', 'speed_diff')

df = pd.read_pickle('df_sample.pkl')
df = df[df['filter'] == 'exp2']
df['speed'] = np.sqrt(df.umean**2+df.vmean**2)
df['speed_track'] = np.sqrt(df.utrack**2+df.vtrack**2)
df['speed_diff'] = df['speed_track']-df['speed']
df = df[['speed', 'speed_diff']].dropna()
df = df[(df.speed <= 20) & (abs(df['speed_diff']) <= 10)]
histogram_plot(df, 'ua', 'speed', 'speed_diff')

df = pd.read_pickle('df_sample.pkl')
df = df[df['filter'] == 'jpl']
df['speed'] = np.sqrt(df.umean**2+df.vmean**2)
df['speed_track'] = np.sqrt(df.utrack**2+df.vtrack**2)
df['speed_diff'] = df['speed_track']-df['speed']
df = df[['speed', 'speed_diff']].dropna()
df = df[(df.speed <= 20) & (abs(df['speed_diff']) <= 10)]
histogram_plot(df, 'jpl', 'speed', 'speed_diff')

df = pd.read_pickle('df_sample.pkl')
df = df[df['filter'] == 'df']
df['speed'] = np.sqrt(df.umean**2+df.vmean**2)
df['speed_track'] = np.sqrt(df.utrack**2+df.vtrack**2)
df['speed_diff'] = df['speed_track']-df['speed']
df = df[['speed', 'speed_diff']].dropna()
df = df[(df.speed <= 20) & (abs(df['speed_diff']) <= 10)]
histogram_plot(df, 'df', 'speed', 'speed_diff')
