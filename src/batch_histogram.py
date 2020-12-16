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
from data import pickled_histograms as ph

pressures = [850, 500]
dts = [3600, 1800]
months = [1, 7]
day_list = (1, 2, 3)
hour_list = (0, 6, 12, 18)


day = 3
hour = 18
for month in months:
    final_triplet = datetime(2006, month, day, hour, 0, 0, 0)

    for dt in dts:
        print('dt: ' + str(dt))
        for pressure in pressures:
            start_time = time.time()
            ds_name = str(dt)+'_' + str(pressure) + '_' + \
                final_triplet.strftime("%B").lower() + '_merged'
            print(ds_name)
           # print('copying starting...')
         #   os.system("rsync   --progress /run/media/amirouyed/reserarchDi/10_03_20/experiments/" +
          #            ds_name + ".nc  ../data/processed/experiments/" + ds_name + ".nc")
            print("copying finished")
           # hist.main(final_triplet, pressure=pressure, dt=dt)
            ph.main(final_triplet, pressure=pressure, dt=dt)
            #ss.main(final_triplet, pressure=pressure, dt=dt)
            # os.system(
            #   "rsync   -progress ../data/processed/dataframes/*.pkl /run/media/amirouyed/reserarchDi/10_03_20/dataframes/")
            #os.system('rm ../data/processed/experiments/*')
            #os.system('rm ../data/processed/dataframes/*')
            print("--- seconds ---" + str(time.time() - start_time))
