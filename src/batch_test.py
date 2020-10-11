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


month = 1
triplet = datetime(2006, month, 3, 18, 0, 0, 0)
pressure = 850
dt = 3600
plots = glob.glob('../data/processed/plots/*')
files = glob.glob('../data/interim/experiments/first_stage_amvs/*')


# os.system("python3 first_stage/first_stage_run.py  -p " + str(pressure) + " -dt 3600   -tri " +
#         triplet.strftime("%Y-%m-%d-%H:%M"))
#ssr.run(triplet, pressure, dt=dt)
#tp.run(triplet, pressure=pressure, dt=dt)
print('triplet: '+str(triplet))
#bp.run(triplet, pressure=pressure, dt=dt)

# print('plotting maps...')


#mm.main(triplet, pressure=pressure, dt=dt)
hist.main(triplet, pressure=pressure, dt=dt)
#ss.main(triplet, pressure=pressure, dt=dt)
#os.system("rsync   --progress  ../data/processed/experiments/*  /run/media/amirouyed/reserarchDi/10_02_20/experiments/")
