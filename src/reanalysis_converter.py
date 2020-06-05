import os
import datetime
import extra_data_analysis as eda
import glob
from pathlib import Path

files = glob.glob('../data/raw/experiments/reanalysis/cfs/*.grb2')

for file in files:
    print(file)
    os.system("wgrib2 " + file + " -netcdf " + Path(file).stem + ".nc")
