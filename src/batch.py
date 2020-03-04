import os


grids = [0.5, 0.25, 0.125, 0.0625]

for grid in grids:
    print('computing for grid: ' + str(grid))
    if grid is not 0.0625:
        os.system(
            "python3 run.py  -a -p -of -fb -b -sd 2006-07-01-05:00 -ed 2006-07-01-07:00 -v 'qv'  -jd -dt 3600 -d -al   -jl  -df -tri 12 -g"+str(grid))
    else:
        os.system(
            "python3 run.py  -a -p -of -fb -b -sd 2006-07-01-05:00 -ed 2006-07-01-07:00 -v 'qv'  -jd -dt 3600 -d -al   -jl  -df -tri 12 ")

    os.system('python3 random_forest.py')
    os.system('cp  errors.txt errorsg'+str(grid)+'.txt')
    os.system('python3 rf_map.py')
    os.system('cp speed_error.png speed_errorg'+str(grid)+'.png')
    os.system('cp speed.png speedg'+str(grid)+'.png')
    os.system('cp stdev.png stdevg'+str(grid)+'.png')
    #os.system('cp speed_error_df.png speed_error_dfg'+str(grid)+'.png')
    #os.system('cp speed_error_jpl.png speed_error_jplg'+str(grid)+'.png')
