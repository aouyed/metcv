import os
import time
import glob
import sh
from datetime import datetime
from second_stage import second_stage_run as ssr
from data import track_preprocessor as tp
from data import batch_plotter as bp
from data import histograms as hist
from data import map_maker as mm
from data import summary_statistics as ss

pressures = [850]
dts = [3600]
#dts = [1800]
triplet_times = []
month = 1
day_list = (1, 2, 3)
hour_list = (0, 6, 12, 18)
# day_list = [1]
# hour_list = [0]
for day in day_list:
    for hour in hour_list:
        triplet_times.append(datetime(2006, month, day, hour, 0, 0, 0))


# print(triplet_times)
files = glob.glob('../data/processed/experiments/200*.nc')
if files:
    sh.rm(files)

plots = glob.glob('../data/processed/plots/*')

if plots:
    sh.rm(plots)

pressure = pressures[0]


for dt in dts:
    for pressure in pressures:
        start_time = time.time()
        for triplet_time in triplet_times:
            os.system("python3 first_stage/first_stage_run.py   -p " + str(pressure) + " -dt " + str(dt) + " -tri " +
                      triplet_time.strftime("%Y-%m-%d-%H:%M"))
            print('done first stage')
            ssr.run(triplet_time, pressure, dt)
            final_triplet = triplet_time
        tp.run(final_triplet, pressure=pressure, dt=dt)
        bp.run(final_triplet, pressure=pressure, dt=dt)
        mm.main(final_triplet, pressure=pressure, dt=dt)
        hist.main(final_triplet, pressure=pressure, dt=dt)
        ss.main(final_triplet, pressure=pressure, dt=dt)
        print("--- seconds ---" + str(time.time() - start_time))
