import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': '850',
        'year': '2006',
        'month': '01',
        'day': '01',
        'time': '00:00',
        'format': 'netcdf',
    },
