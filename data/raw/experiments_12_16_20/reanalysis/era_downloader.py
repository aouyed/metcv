import cdsapi

c = cdsapi.Client()

#var = 'v'
year = '2009'
month = '11'
#day = '04'
#time = '17:00:00'
pressures = ['700']
#timen = '12_00_00'

days = ['04']
times = ['17:00:00']
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
