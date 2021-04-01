import datetime
import os


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

    d1 = datetime.datetime(2006, month, day, 19, 0, 0, 0)
    d0 = d1 - datetime.timedelta(hours=20)
    print('d0 ' + str(d0))
    start_dates = daterange(d0, d1, 6)
    end_dates = daterange(d0+datetime.timedelta(hours=2),
                          d1+datetime.timedelta(hours=2), 6)

    date_list = []
    for i, start_date in enumerate(start_dates):
        #date_list = date_list + daterange(start_date, end_dates[i], 0.5)
        date_list = date_list + daterange(start_date, end_dates[i], 0.5)

    datelists[day] = date_list


for day in [1, 2, 3]:
    dates = datelists[day]

    for date in dates:
        string_date = date.strftime("%Y%m%d_%H%M")
        filename = 'NRData_300hPa_'+string_date+'.nc'
        print(string_date)
        os.system('mv NR300/'+filename +
                  ' ../raw/experiments/jpl/july/0'+str(day)+'/'+filename)
        #filename = 'NRData_300hPa_'+string_date+'.nc'
        # print(string_date)
        # os.system('mv NR300/'+filename +
        #         ' ../raw/experiments/jpl/july/0'+str(day)+' /'+filename)
