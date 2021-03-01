
import pandas as pd

SIGMA_LON = 1.5
SIGMA_LAT = 0.15


def rean_formatter(df):
    df.loc[df['filter'] == 'jpl', 'filter'] = 'JPL'
    df.loc[df['filter'] == 'rean', 'filter'] = 'Reanalysis'
    df = df.rename(columns={'rmse': 'RMSVD [m/s]', 'filter': 'Algorithm'})
    df = df[['Algorithm', 'RMSVD [m/s]']]
    print(df)
    return(df)


def sensitivity_formatter(df, with_fsua):
    df = df.loc[df['exp_filter'] == 'rf'].copy()
    df = df.loc[df['with_fsua'] == with_fsua]
    df['sigma_lat'] = df['factor']*SIGMA_LAT
    df['sigma_lon'] = df['factor']*SIGMA_LON
    df = df[['sigma_lat', 'sigma_lon', 'rmse']]
    df = df.rename(columns={
                   'rmse': 'RMSVD [m/s]', 'sigma_lat': '$\sigma_{\phi}$', 'sigma_lon': '$\sigma_{\lambda}$'})
    print(df)
    return(df)


def loop(myfile, df):
    myfile.write('\\begin{tabular}{c} \n')
    myfile.write('(a)')
    myfile.write('\\centering\n')
    myfile.write(df.round(2).to_latex(index=False, escape=False))
    myfile.write('\end{tabular} \\\\ \n')


df = pd.read_pickle('rean_summary.pkl')
myfile = open('sensitivity_stats.txt', 'w')

df = pd.read_pickle('rean_summary.pkl')
myfile = open('sensitivity_stats.txt', 'w')


df = rean_formatter(df)
loop(myfile, df)
df = pd.read_pickle('sensitivity.pkl')
df = sensitivity_formatter(df, True)
loop(myfile, df)
df = pd.read_pickle('sensitivity.pkl')
df = sensitivity_formatter(df, False)
loop(myfile, df)


# loop(myfile, table_quantile)
myfile.close()
# myfile = open('all_stats_stats.txt', 'r')

# data = myfile.read()
# data = data.replace('{lrrr}', '{|c|c|c|c|}')
# data = data.replace('{lrrrr}', '{|c|c|c|c|c|}')
# data = data.replace('\\midrule', '\\hline')
# data = data.replace('\\toprule', '\\hline')
# data = data.replace('\\bottomrule', '\\hline')

# myfile.close()
# myfile = open('all_stats_stats.txt', 'w')
# myfile.write(data)
# myfile.close()
