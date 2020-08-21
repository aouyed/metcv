import os
import glob
import sh
from datetime import datetime
from second_stage import second_stage_run as ssr
from data import track_preprocessor as tp
from data import batch_plotter as bp
from data import histograms as hist
from data import map_maker as mm
from data import summary_statistics as ss


triplet = datetime(2006, 1, 3, 18, 0, 0, 0)
pressure = 850
dt = 3600
files = glob.glob('../data/processed/experiments/2006*')
plots = glob.glob('../data/processed/plots/*')

# if plots:
#   sh.rm(plots)

# os.system("python3 first_stage/first_stage_run.py  -p " + str(pressure) + " -dt 3600   -tri " +
#         triplet.strftime("%Y-%m-%d-%H:%M"))
#ssr.run(triplet, pressure, dt=dt)
#tp.run(triplet, pressure=pressure, dt=dt)
#bp.run(triplet, pressure=pressure, dt=dt)

print('plotting maps...')

#mm.main(triplet, pressure=pressure, dt=dt)
#hist.main(triplet, pressure=pressure, dt=dt)
ss.main(triplet, pressure=pressure, dt=dt)
