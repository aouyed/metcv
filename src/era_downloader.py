import cdsapi

c = cdsapi.Client()

#var = 'v'
year = '2006'
month = '07'
day = '01'
time = '12:00:00'
pressure = '850'
timen = '12_00_00'

varis = ['u', 'v']
for var in varis:
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
        var + '_'+pressure + '_'+year + '_'+month+'_'+day + '_'+timen + '_era5.nc')
