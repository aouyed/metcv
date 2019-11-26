#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:37:42 2019

@author: aouyed
"""

import xarray as xr
import requests
import netCDF4
import boto3
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import metpy 
import cartopy
import seaborn as sns
import cartopy.crs as ccrs
from shapely import geometry
from mpl_toolkits.basemap import Basemap


#from mpl_toolkits.basemap import Basemap








def get_s3_keys(bucket, prefix = ''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    """
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket}

    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix):
                yield key

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break
        
bucket_name = 'noaa-goes16'
#product_name = 'ABI-L1b-RadF'
product_name = 'ABI-L2-CMIPF'
year = 2019
day_of_year = 79
hour = 12
band = 11




def getImage(hour, bands):
    for band in bands:
        print(band)
        keys = get_s3_keys(bucket_name, prefix = product_name+'/'+ 
                                             str(year) + '/' + str(day_of_year).zfill(3) 
                                             + '/' + str(hour).zfill(2) + '/OR_'+ 
                                             product_name + '-M3C' + str(band).zfill(2))
    
        
        
        #selecting intervals of 15 minutes 
        keys = np.array([key for key in keys])
        a=len(keys)/4-1
        b=2*len(keys)/4-1
        c=3*len(keys)/4-1
        d=len(keys)-1
        e=len(keys)/2
        ind_pos=[0,int(e)]
        keys = keys[ind_pos] 
        
        k=0
        for key in keys: 
            resp = requests.get('https://' + bucket_name + '.s3.amazonaws.com/' + key)
            
            file_name = key.split('/')[-1].split('.')[0]
            print(file_name)
            nc4_ds = netCDF4.Dataset(file_name, memory = resp.content)
            store = xr.backends.NetCDF4DataStore(nc4_ds)
            DS = xr.open_dataset(store)
            print(DS.data_vars)
            dat=DS.metpy.parse_cf('CMI')
            geos = dat.metpy.cartopy_crs
            x = dat.x
            y = dat.y

            ##############################
            plt.imsave('amv_lossless/'+str(band)+'_'+str(hour)+"_"+str(k)+'_'+file_name + '.png',DS.CMI.fillna(0), cmap='gray')
            plt.close()
            np.savetxt('image_arrays/'+str(band)+'_'+str(hour)+"_"+str(k)+'_'+file_name + '.txt',DS.CMI.fillna(0))
            np.savetxt('coordinate_arrays/x_'+str(band)+'_'+str(hour)+"_"+str(k)+'_'+file_name + '.txt',DS.x.fillna(0))
            np.savetxt('coordinate_arrays/y_'+str(band)+'_'+str(hour)+"_"+str(k)+'_'+file_name + '.txt',DS.y.fillna(0))

            ###########################
            bmap = Basemap(projection='geos', lon_0=-89.5, lat_0=0.0, satellite_height=35786023.0, ellps='GRS80')
            bmap.imshow(DS.CMI.fillna(0),vmin=170, vmax=378, origin='upper', cmap='Greys')
            bmap.drawcoastlines(linewidth=0.3, linestyle='solid', color='black')
            bmap.drawcountries(linewidth=0.3, linestyle='solid', color='black')
            bmap.drawparallels(np.arange(-90.0, 90.0, 10.0), linewidth=0.1, color='black')
            bmap.drawmeridians(np.arange(0.0, 360.0, 10.0), linewidth=0.1, color='black')
             
            # Insert the legend
            bmap.colorbar(location='bottom', label='Brightness Temperature [K]')
             
            # Export result
            DPI = 300

            plt.savefig('amv_images/projection_'+str(hour)+"_"+str(k)+'_'+file_name + '.png', dpi=DPI, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()
            k=k+1
 


hours=np.arange(1,24,1)
bands=[8,9,10]
bands=[11]
for hour in hours:
    print(hour)
    getImage(hour,bands)
    



print("Done!")