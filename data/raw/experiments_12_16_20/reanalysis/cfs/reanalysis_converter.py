import os
import datetime
import glob
from pathlib import Path

files = glob.glob('pgbhnl*.grb2')

for file in files:
    print(file)
    os.system("wgrib2 " + file + " -netcdf " + Path(file).stem + ".nc")
