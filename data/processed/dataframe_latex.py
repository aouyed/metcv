import pandas as pd
import glob

# files = glob.glob('*jan*stats.pkl')
files = glob.glob('dataframes/*df_stats.pkl')
end_file = files[-1]


def table_skewness(df, myfile, dt, pressure):
    dt = int(int(dt)/60)
    dt = str(dt)
    df = df[df['var'] == 'speed']
    df = df[['filter', 'skewness', 'stdev', 'mean']]

    df = df.rename(columns={"filter": "Algorithm",
                            "stdev": "Standard Deviation [m/s]", "skewness": "Skewness", "mean": "Mean [m/s]"}, errors="raise")
    # df = df.rename(columns={"q50": "50 %", "q68": "68 %",
    #                        "q95": "95 %"}, errors="raise")
    myfile.write('\\'+'begin{subtable}{0.7\\textwidth}' + '\n')
    myfile.write('\\centering'+'\n')
    myfile.write(df.round(2).to_latex(index=False))
    myfile.write('\\caption{$\\Delta t='+dt +
                 '$ min, $P =' + pressure + '$ hPa}' + '\n')
    myfile.write('\end{subtable}' + '\n')


def table_quantile(df, myfile, dt, pressure):
    dt = int(int(dt)/60)
    dt = str(dt)
    df = df[df['var'] == 'angle']
    df = df[['filter', 'q50', 'q68', 'q95', 'q99']]

    df = df.rename(columns={"filter": "Algorithm", "q50": "50 %", "q68": "68 %",
                            "q95": "95 %", "q99": "99 %"}, errors="raise")
    myfile.write('\\'+'begin{subtable}{0.7\\textwidth}' + '\n')
    myfile.write('\\centering'+'\n')
    myfile.write(df.round(2).to_latex(index=False))
    myfile.write('\\caption{$\\Delta t='+dt +
                 '$ min, $P =' + pressure + '$ hPa}' + '\n')
    myfile.write('\end{subtable}' + '\n')


def loop(myfile, table_func):
    myfile.write('\\begin{table}'+'\n')
    myfile.write('\\centering'+'\n')

    for month in ['january', 'july']:
        if month == 'july':
            myfile.write('\end{table}'+'\n')
            myfile.write(
                month + '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
            myfile.write('\\begin{table}'+'\n')
            myfile.write('\\centering'+'\n')
        for dt in ['3600', '1800']:
            for pressure in ['850', '500']:
                filename = 'dataframes/' + dt+'_'+month+'_' + pressure + '_df_stats.pkl'
                df = pd.read_pickle(filename)
                print(df)
                df = df[df['filter'] != 'reanalysis']
                df['filter'][df['filter'] == 'exp2'] = 'UA'
                df['filter'][df['filter'] == 'df'] = 'fsUA'
                df['filter'][df['filter'] == 'jpl'] = 'JPL'
                # df['var'][df['var'] == 'speed'] = 'All'
                # df['var'][df['var'] == 'angle'] = 'angle = 90 degrees'
                myfile.write('%'+filename + '\n')
                table_func(df.copy(), myfile, dt, pressure)
    myfile.write('\end{table}'+'\n')


myfile = open('all_stats_stats.txt', 'w')
loop(myfile, table_skewness)
loop(myfile, table_quantile)
myfile.close()
myfile = open('all_stats_stats.txt', 'r')

data = myfile.read()
data = data.replace('{lrrr}', '{|c|c|c|c|}')
data = data.replace('{lrrrr}', '{|c|c|c|c|c|}')
data = data.replace('\\midrule', '\\hline')
data = data.replace('\\toprule', '\\hline')
data = data.replace('\\bottomrule', '\\hline')

myfile.close()
myfile = open('all_stats_stats.txt', 'w')
myfile.write(data)
myfile.close()
