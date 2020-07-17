import os
import glob
import sh
from datetime import datetime
from second_stage import second_stage_run as ssr
from data import track_preprocessor as tp
from data import batch_plotter as bp
from data import histograms as hist
from data import map_maker as mm

# triplet_times = (datetime(2006, 7, 1, 0, 0, 0, 0), datetime(2006, 7, 1, 6, 0, 0, 0),
#                datetime(2006, 7, 1, 12, 0, 0, 0), datetime(2006, 7, 1, 18, 0, 0, 0))

pressure = 850
dt = 1800
files = glob.glob('../data/processed/experiments/2006*')
# if files:
#   sh.rm(files)
#triplet_time = datetime(2006, 7, 1, 0, 0, 0, 0)
# os.system("python3 first_stage/first_stage_run.py  -p " + str(pressure) + " -dt 3600   -tri " +
#         triplet_time.strftime("%Y-%m-%d-%H:%M"))
#ssr.run(triplet_time, pressure)
# tp.run(pressure)
#mm.main(pressure=pressure, dt=dt)

#bp.run(pressure, dt)

print('plotting maps...')
#hist.main(pressure=pressure, dt=dt)
mm.main(pressure=pressure, dt=dt)
