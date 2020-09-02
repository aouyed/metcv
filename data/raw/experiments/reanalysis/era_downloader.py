import cdsapi

c = cdsapi.Client()

#var = 'v'
year = '2006'
month = '07'
#day = '01'
#time = '12:00:00'
pressures = ('500', '850')
#timen = '12_00_00'

days = ['01', '02', '03']
times = ['00:00:00', '06:00:00', '12:00:00', '18:00:00']
vars = ['u', 'v']

for pressure in pressures:
    for day in days:
        for time in times:
            for var in vars:
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': var,
                        'pressure_level': pressure,
                        'year': year,
                        'month': month,
                        'day': day,
                        'time': time,
                        'format': 'netcdf',                 # Supported format: grib and netcdf. Default: grib
                    },
                    var + '_'+pressure + '_'+year + '_'+month+'_'+day + '_'+time + '_era5.nc')
