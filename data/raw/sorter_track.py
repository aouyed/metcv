import datetime
import os
import xarray as xr


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = datetime.timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list


month = 7
day = 1
datelists = {}
for day in [1, 2, 3]:

    d1 = datetime.datetime(2006, month, day, 18, 0, 0, 0)
    d0 = datetime.datetime(2006, month, day, 0, 0, 0, 0)
    print('d0 ' + str(d0))
    date_list = daterange(d0, d1, 6)

    datelists[day] = date_list


for day in [1]:
    dates = datelists[day]

    for date in dates:
        string_date = date.strftime("%Y%m%d_%H%M")
        #filename = 'NRData_300hPa_'+string_date+'.nc'
        # print(string_date)
        # os.system('mv NR300/'+filename +
        #         ' ../raw/experiments/jpl/january/0'+str(day)+'/'+filename)
        filename = 'NRTracked_3img_300hPa_dt2_sig8.4_'+string_date+'.nc'
        print(string_date)
        os.system('mv tracked_wind/'+filename +
                  ' ../raw/experiments/jpl/tracked/july/'+str(day) + '/300/60min/'+filename)
