import os
import glob
import sh
from datetime import datetime
from ml import extra_data_analysis as eda
from data import track_preprocessor as tp
from data import batch_plotter as bp

triplet_times = []
month = 7
day_list = (1, 2, 3)
hour_list = (0, 6, 12, 18)
#day_list = [1]
#hour_list = [0]
for day in day_list:
    for hour in hour_list:
        triplet_times.append(datetime(2006, month, day, hour, 0, 0, 0))


print(triplet_times)
files = glob.glob('../data/processed/experiments/*')
if files:
    sh.rm(files)
for triplet_time in triplet_times:
    print('loop')
    os.system("python3 run.py  -p 500  -dt 3600  -tri " +
              triplet_time.strftime("%Y-%m-%d-%H:%M"))
    eda.run(triplet_time)

tp.run()
bp.run()
