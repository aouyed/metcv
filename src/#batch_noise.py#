import os


noises = [0.1]
for noise in noises:
    print('computing for noise: ' + str(noise))
    os.system("python3 run.py  -a -p -of -fb -b -sd 2006-07-01-05:00 -ed 2006-07-01-07:00 -v 'qv'  -jd -dt 3600 -d -al   -jl  -df -tri 12 -sr " + str(noise))

    os.system('python3 random_forest.py')
    os.system('cp  errors.txt errorsn'+str(noise)+'.txt')
    os.system('python3 rf_map.py')
    os.system('cp speed_error.png speed_errorn'+str(noise)+'.png')
    os.system('cp speed.png speedn'+str(noise)+'.png')
    os.system('cp stdev.png stdevn'+str(noise)+'.png')
    # os.system('cp speed_error_df.png speed_error_dfg'+str(grid)+'.png')
    # os.system('cp speed_error_jpl.png speed_error_jplg'+str(grid)+'.png')
