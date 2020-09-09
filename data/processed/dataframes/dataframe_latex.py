import pandas as pd
import glob

files = glob.glob('*jan*stats.pkl')
end_file = files[-1]

myfile = open('jan_stats.txt', 'w')

myfile.write('\\begin{table}'+'\n')
myfile.write('\\centering'+'\n')
for file in files:
    df = pd.read_pickle(file)
    df = df[df['filter'] != 'reanalysis']
    df = df[df['var'] == 'speed']
    df = df[['filter', 'skewness', 'stdev', 'mean']]
    df['filter'][df['filter'] == 'exp2'] = 'UA'
    df['filter'][df['filter'] == 'df'] = 'fsUA'
    df['filter'][df['filter'] == 'jpl'] = 'JPL'
    df = df.rename(columns={"filter": "Algorithm",
                            "stdev": "Standard Deviation [m/s]", "skewness": "Skewness", "mean": "Mean [m/s]"}, errors="raise")
    myfile.write('\\'+'begin{subtable}{0.7\\textwidth}' + '\n')
    myfile.write(df.round(2).to_latex(index=False))
    myfile.write('\caption{'+file+'}' + '\n')
    myfile.write('\end{subtable}' + '\n')
    if file != end_file:
        myfile.write('\quad' + '\n')


myfile.write('\end{table}'+'\n')
myfile.close()
