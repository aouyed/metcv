import pandas as pd
import glob

files = glob.glob('*stats.pkl')

for file in files:
    df = pd.read_pickle(file)
    myfile = open('stats.txt', 'w')
    myfile.write(df.round(2).to_latex() + '\n')

myfile.close()
