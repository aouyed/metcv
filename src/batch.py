import os
import datetime
import extra_data_analysis as eda

triplet_time = datetime.datetime(2006, 7, 1, 0, 0, 0, 0)


# os.system("python3 run.py  -a -p  -v 'qv' -dt 3600  -df -tri " +
#         triplet_time.strftime("%Y-%m-%d-%H:%M"))
eda.run(triplet_time)
