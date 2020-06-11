import os
import glob
import sh
from datetime import datetime
import extra_data_analysis as eda
import track_preprocessor as tp
import batch_plotter as bp
# triplet_times = (datetime(2006, 7, 1, 0, 0, 0, 0), datetime(2006, 7, 1, 6, 0, 0, 0),
#                datetime(2006, 7, 1, 12, 0, 0, 0), datetime(2006, 7, 1, 18, 0, 0, 0))


#files = glob.glob('../data/processed/experiments/*')
# sh.rm(files)
#triplet_times = [datetime(2006, 7, 1, 0, 0, 0, 0)]
# for triplet_time in triplet_times:
#  os.system("python3 run.py  -a -p  -v 'qv' -dt 3600  -df -tri " +
#           triplet_time.strftime("%Y-%m-%d-%H:%M"))
# eda.run(triplet_time)

# tp.run()
bp.run()
