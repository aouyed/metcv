
from joblib import dump, load
from viz import amv_analysis as aa
from viz import dataframe_calculators as dfc
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
from sklearn.model_selection import train_test_split

sns.set_style('white', {'legend.frameon': None})
# mpl.rcParams['figure.dpi'] = 150
# sns.set_context("paper")
sns.set_context('talk')
pd.set_option('display.expand_frame_repr', False)
dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


def shaded_plot(x, y, err, ax, co):
    ax.plot(x, y, color=co)
    ax.fill_between(x, y-err, y+err, alpha=0.6, color=co)


def plotter(df, values):
    grid = 10
    piv = df.pivot('y', 'x', values).values
    U = df.pivot('y', 'x', 'u').values
    V = df.pivot('y', 'x', 'v').values

    factor = 0.0625/grid

    U = cv2.resize(U, None, fx=factor, fy=factor)
    V = cv2.resize(V, None, fx=factor, fy=factor)
    print(U.shape)
    print(V.shape)
    X = np.arange(-180, 180 - grid, grid)
    Y = np.arange(-90, 90 - grid, grid)
    print(len(X))
    print(len(Y))
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    Q = ax.quiver(X, Y, U, V, scale=250)
    qk = plt.quiverkey(Q, 0.5, 0.5, 2, r'$2 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
    pmap = plt.cm.BuGn
    pmap.set_bad(color='black')
    im = ax.imshow(piv, cmap=pmap, extent=[-180, 180, -90, 90], origin='lower')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    cbar.set_label('g/kg')
    plt.xlabel("lon")
    plt.ylabel("lat")
    ax.set_title('Truth velocities and absolute humidity')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def plotter_res(df,  values, title):

    df['speed_error'] = np.sqrt(df['speed_error'])
    grid = 10
    speed_error = df.pivot('lat', 'lon', 'speed_error').values

    factor = 0.0625/grid

    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    pmap = plt.cm.RdPu
    pmap.set_bad(color='black')
    im = ax.imshow(speed_error, cmap=pmap,
                   extent=[-180, 180, -90, 90], origin='lower', vmin=0, vmax=3)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    cbar.set_label('m/s ')
    plt.xlabel("lon")
    plt.ylabel("lat")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def scatter_plotter(X, values):
    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    ax.scatter(X['lon'], X['lat'], s=0.1)
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title('Training Data')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def line_plotter(X, X_2, X_3, X_4, values):
    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    sns.lineplot(X['speed'], X['speed_approx'], label='jpl', ax=ax)
    sns.lineplot(X_2['speed'], X_2['speed_approx'], label='physics', ax=ax)
    sns.lineplot(X_3['speed'], X_3['speed_approx'], label='vem', ax=ax)
    sns.lineplot(X_4['speed'], X_4['speed_approx'], label='physics_qv', ax=ax)

    sns.lineplot(X['speed'], X['speed'], label='truth', ax=ax)
    # ax.legend()

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("ground truth [m/s]")
    ax.set_ylabel("AMV [m/s]")
    ax.set_title('Wind Speeds')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    sns.lineplot(X['speed'], X['speed_approx_std'], label='jpl', ax=ax)
    sns.lineplot(X_2['speed'], X_2['speed_approx_std'], label='physics', ax=ax)
    sns.lineplot(X_3['speed'], X_3['speed_approx_std'], label='vem', ax=ax)
    sns.lineplot(X_4['speed'], X_4['speed_approx_std'],
                 label='physics_qv', ax=ax)
    # ax.legend()

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("ground truth [m/s]")
    ax.set_ylabel("stdev [m/s]")
    ax.set_title('weighted standard deviations')
    directory = '../data/processed/density_plots'
    plt.savefig('stdev.png', bbox_inches='tight', dpi=300)


def line_plotter_1(X, X_2, X_3, X_4,  values):
    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    sns.lineplot(X['speed'], X['speed_error'], label='jpl', ax=ax)
    sns.lineplot(X_2['speed'], X_2['speed_error'], label='physics', ax=ax)
    sns.lineplot(X_3['speed'], X_3['speed_error'], label='vem', ax=ax)
    sns.lineplot(X_4['speed'], X_4['speed_error'], label='physics_qv', ax=ax)

    # sns.lineplot(X['speed'], X['speed'], label='truth', ax=ax)
    # ax.legend()

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("ground truth [m/s]")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title('RMSVD')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def line_plotter_0(df, values):
    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()

    sns.lineplot(df['hours'], df['jpl'], label='jpl',
                 linestyle='--', marker='o', ax=ax)
    sns.lineplot(df['hours'], df['df'], label='vem',
                 linestyle='--', marker='o', ax=ax)
    sns.lineplot(df['hours'], df['rf'], label='physics',
                 linestyle='--', marker='o', ax=ax)

    ax.legend(frameon=None)

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                 linewidth=2, color='gray', alpha=0, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    ax.set_xlabel("hour")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title('Results')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


def averager(df, xlistv):
    df_mean = dfc.plot_average(
        deltax=1, df=df, xlist=xlistv, varx='speed', vary='speed_approx')
    df_mean_e = dfc.plot_average(
        deltax=1, df=df, xlist=xlistv, varx='speed', vary='speed_error')
    df_mean_e['speed_error'] = np.sqrt(df_mean_e['speed_error'])
    return df_mean, df_mean_e


def X_test_init(X_test, regressor):
    print('predicting...')
    y_pred = regressor.predict(X_test)
    X_test['speed_approx'] = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
    X_test['speed_error'] = (y_pred[:, 0]-y_test['u'])**2 + \
        (y_pred[:, 1]-y_test['v'])**2
    X_test['speed'] = np.sqrt(y_test['u']**2 + y_test['v']**2)
    X_test['cos_weight'] = np.cos(X_test['lat']/180*np.pi)
    return X_test, y_pred

    # dump(regressor, 'rf.joblib')
regressor = load('rf_uv.joblib')
y_test = load('yte_uv.joblib')
X_test = load('xte_uv.joblib')

regressor_qv = load('rf_qv.joblib')
y_test_qv = load('yte_qv.joblib')
X_test_qv = load('xte_qv.joblib')

regressor_dqv = load('rf_dqv.joblib')
y_test_dqv = load('yte_dqv.joblib')
X_test_dqv = load('xte_dqv.joblib')

hours = [0, 6, 12, 18]
rf = [1.7122288824810241, 1.6554113125780026,
      1.614328436366025, 1.7043883044378514]
deepf = [3.0814844441258487, 2.8653066164477585,
         2.7922954295197595, 2.9260425156666905]
jpl = [3.7132896180961383, 3.412857290296869,
       3.363429985554871, 3.440039715662562]


d = {'hours': hours, 'df': deepf, 'jpl': jpl, 'rf': rf}
df_results = pd.DataFrame(data=d)
print(df_results)

start_date = datetime.datetime(2006, 7, 1, 6, 0, 0, 0)
end_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)
df = aa.df_concatenator(dataframes_dict, start_date,
                        end_date, False, True, False)

df = df.dropna()

# df['qv'] = 1000*df['qv']
# plotter(df, 'qv')


X_test, y_pred = X_test_init(X_test, regressor)
X_test_qv, y_pred_qv = X_test_init(X_test_qv, regressor_qv)
X_test_dqv, y_pred_dqv = X_test_init(X_test_dqv, regressor_dqv)

df_jpl = aa.df_concatenator(dataframes_dict, start_date,
                            end_date, True, True, False)


plotter_res(df_jpl.copy(), 'speed_error_jpl', 'jpl')

xlistv = np.arange(df['speed'].min(), df['speed'].max(), 1)
print('averaging...')

df_mean, df_mean_e = averager(df_jpl, xlistv)
df_mean_vem, df_mean_vem_e = averager(df, xlistv)
df_mean_rf, df_mean_rf_e = averager(X_test, xlistv)
df_mean_rf_qv, df_mean_rf_qv_e = averager(X_test_qv, xlistv)
df_mean_rf_dqv, df_mean_rf_dqv_e = averager(X_test_qv, xlistv)

plotter_res(df_jpl.copy(), 'speed_error_jpl', 'jpl')
plotter_res(X_test.copy(), 'speed_error_rf', 'physics')
plotter_res(X_test_qv.copy(), 'speed_error_rf', 'physics_qv')


print('plotting...')
line_plotter(df_mean, df_mean_rf, df_mean_vem, df_mean_rf_qv, 'speed')
line_plotter_1(df_mean_e, df_mean_rf_e, df_mean_vem_e,
               df_mean_rf_qv_e, 'speed_error')

line_plotter_0(df_results, 'results')
