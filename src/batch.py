import os
import glob
import sh
from datetime import datetime
import extra_data_analysis as eda
import track_preprocessor as tp
import batch_plotter as bp
# triplet_times = (datetime(2006, 7, 1, 0, 0, 0, 0), datetime(2006, 7, 1, 6, 0, 0, 0),
#                datetime(2006, 7, 1, 12, 0, 0, 0), datetime(2006, 7, 1, 18, 0, 0, 0))

triplet_times = []
month = 7
for day in (1, 2, 3):
    for hour in (0, 6, 12, 18):
        triplet_times.append(datetime(2006, month, day, hour, 0, 0, 0))


print(triplet_times)
files = glob.glob('../data/processed/experiments/*')
if files:
    sh.rm(files)
for triplet_time in triplet_times:
    os.system("python3 run.py  -a -p  -v 'qv' -dt 3600  -df -tri " +
              triplet_time.strftime("%Y-%m-%d-%H:%M"))
    eda.run(triplet_time)

tp.run()
bp.run()
