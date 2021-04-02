import pandas as pd
import glob as glob
import numpy as np

files = glob.glob('../sampler_test*')

rmses = []
for file in files:
    df = pd.read_csv(file)
    val = df.loc[df['exp_filter'] == 'rf', 'rmse'].values
    rmses.append(val[0])
rmses = np.array(rmses)
print(rmses.mean())
print(rmses.std())
