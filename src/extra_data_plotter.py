import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def line_plotter(df0, values):
    fig, ax = plt.subplots()

    df = df0[df0.categories == 'poly']
    ax.plot(np.array(df['latlon']), df['rmse'], '-o', label='poly')

    df = df0[df0.categories == 'rf']
    ax.plot(df['latlon'], df['rmse'], '-o', label='rf')

    df = df0[df0.categories == 'df']
    ax.plot(df['latlon'], df['rmse'], '-o', label='vem')

    df = df0[df0.categories == 'jpl']
    ax.plot(df['latlon'], df['rmse'], '-o', label='jpl')

    ax.legend(frameon=None)

    ax.set_xlabel("Region")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title('Results')
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)


df0 = pd.read_pickle("./df_results.pkl")
df0.sort_values(by=['latlon'], inplace=True)
df = df0[df0.extra == True]
line_plotter(df, 'results_e')
df = df0[df0.extra == False]
line_plotter(df, 'results')
