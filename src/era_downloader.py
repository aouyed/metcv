import cdsapi

c = cdsapi.Client()

var = 'v'
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': var,
        'pressure_level': '850',
        'year': '2006',
        'month': '01',
        'day': '01',
        'time': '12:00:00',
        'format': 'netcdf',                 # Supported format: grib and netcdf. Default: grib
    },
    var + '_era5.nc')
