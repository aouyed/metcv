import os
import glob
import sh
from datetime import datetime
from second_stage import second_stage_run as ssr
from data import track_preprocessor as tp
from data import batch_plotter as bp
from data import histograms as hist
# triplet_times = (datetime(2006, 7, 1, 0, 0, 0, 0), datetime(2006, 7, 1, 6, 0, 0, 0),
#                datetime(2006, 7, 1, 12, 0, 0, 0), datetime(2006, 7, 1, 18, 0, 0, 0))

pressure = 500
files = glob.glob('../data/processed/experiments/*')
# if files:
#   sh.rm(files)
#triplet_times = [datetime(2006, 7, 1, 0, 0, 0, 0)]
# for triplet_time in triplet_times:
#    os.system("python3 first_stage/first_stage_run.py  -p " + str(pressure) + " -dt 3600   -tri " +
#             triplet_time.strftime("%Y-%m-%d-%H:%M"))
#  ssr.run(triplet_time)

# tp.run(pressure)
bp.run(pressure)
hist.main(pressure=pressure)
