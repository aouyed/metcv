import os
import datetime
import glob
from pathlib import Path

files = glob.glob('pgbh00*')

for file in files:
    print(file)
    os.system("wgrib2 " + file + " -netcdf " + Path(file).stem + ".nc")
