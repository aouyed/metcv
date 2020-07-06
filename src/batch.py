import os
import glob
import sh
from datetime import datetime
from second_stage import second_stage_run as ssr
from data import track_preprocessor as tp
from data import batch_plotter as bp
from data import histograms as hist
pressures = [850]
triplet_times = []
month = 7
day_list = (1, 2, 3)
hour_list = (0, 6, 12, 18)
# day_list = [1]
# hour_list = [0]
for day in day_list:
    for hour in hour_list:
        triplet_times.append(datetime(2006, month, day, hour, 0, 0, 0))


# print(triplet_times)
files = glob.glob('../data/processed/experiments/*')
if files:
    sh.rm(files)
for pressure in pressures:
    for triplet_time in triplet_times:
        os.system("python3 first_stage/first_stage_run.py   -p " + str(pressure) + " -dt 3600  -tri " +
                  triplet_time.strftime("%Y-%m-%d-%H:%M"))
        ssr.run(triplet_time)
    tp.run(pressure)
    bp.run(pressure)
    hist.main(pressure=pressure)
