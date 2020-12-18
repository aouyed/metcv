import os
import glob
import sh
from datetime import datetime
from second_stage import second_stage_run as ssr
from data import track_preprocessor as tp
from data import batch_plotter as bp
from data import histograms as hist
from data import map_maker_better as mm
from data import summary_statistics as ss
from data import pickled_results as pr
from data import stdev_stats as sds
from data import pickled_histograms as ph

month = 11
triplet = datetime(2009, month, 4, 17, 0, 0, 0)
pressure = 700
# dt = 3600/4
dt = 900
plots = glob.glob('../data/processed/plots/*')
files = glob.glob('../data/interim/experiments/first_stage_amvs/*')

# if files:
#  sh.rm(files)

# os.system("python3 first_stage/first_stage_run.py  -p " + str(pressure) +
#         " -dt " + str(dt) + "  -tri " + triplet.strftime("%Y-%m-%d-%H:%M"))
ssr.run(triplet, pressure, dt=dt)
# tp.run(triplet, pressure=pressure, dt=dt)
print('triplet: '+str(triplet))
# bp.run(triplet, pressure=pressure, dt=dt)

# print('plotting maps...')

# mm.main(triplet, pressure=pressure, dt=dt)
# hist.main(triplet, pressure=pressure, dt=dt)
# ph.main(triplet, pressure=pressure, dt=dt)

# ss.main(triplet, pressure=pressure, dt=dt)
# os.system("rsync   --progress  ../data/processed/experiments/*  /run/media/amirouyed/reserarchDi/10_02_20/experiments/")


# pr.main()
# sds.main()
