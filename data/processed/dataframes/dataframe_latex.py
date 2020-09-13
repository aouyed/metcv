import pandas as pd
import glob

#files = glob.glob('*jan*stats.pkl')
files = ['3600_january_850_df_stats.pkl']
end_file = files[-1]

myfile = open('jan_stats.txt', 'w')

myfile.write('\\begin{table}'+'\n')
myfile.write('\\centering'+'\n')
for file in files:
    df = pd.read_pickle(file)
    print(df)
    df = df[df['filter'] != 'reanalysis']
    df = df[['var', 'filter', 'skewness', 'stdev', 'mean', 'q50', 'q68', 'q95']]
    df['filter'][df['filter'] == 'exp2'] = 'UA'
    df['filter'][df['filter'] == 'df'] = 'fsUA'
    df['filter'][df['filter'] == 'jpl'] = 'JPL'
    df['var'][df['var'] == 'speed'] = 'All'
    df['var'][df['var'] == 'angle'] = 'angle = 90 degrees'
    df = df.rename(columns={'var': ' ', "filter": "Algorithm",
                            "stdev": "Standard Deviation [m/s]", "skewness": "Skewness", "mean": "Mean [m/s]"}, errors="raise")
    df = df.rename(columns={"q50": "50 %", "q68": "68 %",
                            "q95": "95 %"}, errors="raise")
    myfile.write('\\'+'begin{subtable}{0.7\\textwidth}' + '\n')
    myfile.write(df.round(2).to_latex(index=False))
    myfile.write('\caption{'+file+'}' + '\n')
    myfile.write('\end{subtable}' + '\n')
    if file != end_file:
        myfile.write('\quad' + '\n')


myfile.write('\end{table}'+'\n')
myfile.close()
