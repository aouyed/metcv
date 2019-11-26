#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:49:43 2019

@author: amirouyed
"""
import cv2
import numpy as np
import glob
from natsort import natsorted 
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import os

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from mpl_toolkits.basemap import Basemap


def temperatureThresholder(image):
    thresh=image
    thresh[thresh > 233] = 255
    thresh[thresh == 0] = 255
    thresh[thresh <= 233] = 0
    print(np.count_nonzero(thresh==0))
    thresh=thresh.astype(np.uint8)
    return thresh

def contourMaker(thresh):
    edged=cv2.Canny(thresh,0,255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dilated = cv2.dilate(edged, kernel)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('Numbers of contours found=' + str(len(contours)))
    return contours,edged
    
def areaMaker(contours):
    areas=[]
    for c in contours:
        area=cv2.contourArea(c)
        areas.append(area)
    areas=np.array(areas)            
    print(areas.shape)
    areas=areas[areas>1250]
    print(areas.shape)
    

def areaCuller(contours,filename):
    culled_contours=[]
    boundRect=[]
    contours_poly=[]
    for i in range(0,len(contours)):
        if cv2.contourArea(contours[i])>=2500:
            culled_contours.append(contours[i])
            poly=cv2.approxPolyDP(contours[i], 3, True)
            contours_poly.append(poly)
            boundRect.append(cv2.boundingRect(poly))
    np.save(''.join(['contours_culled/',filename,'_rectangles.npy']),boundRect)
    print('Numbers of contours found=' + str(len(culled_contours)))
    
    return(culled_contours)
    
def mapProjector(image, option,filename):
    bmap = Basemap(projection='geos', lon_0=-89.5, lat_0=0.0, satellite_height=35786023.0, ellps='GRS80')
    bmap.imshow(image,origin='upper', cmap='gray')
    bmap.drawcoastlines(linewidth=0.3, linestyle='solid', color='black')
    bmap.drawcountries(linewidth=0.3, linestyle='solid', color='black')
    bmap.drawparallels(np.arange(-90.0, 90.0, 10.0), linewidth=1, color='black')
    bmap.drawmeridians(np.arange(0.0, 360.0, 10.0), linewidth=1, color='black')
    bmap.colorbar(location='bottom', label='Brightness Temperature [K]')
    plt.title('T<233 K spots, March 2019, 8.4 microns '+option)

    plt.savefig('mcs_images/'+filename+'_'+option+'_bmap.png', dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


filelist=natsorted(glob.glob('image_arrays/11_*'))

for file in filelist:
    cv2.destroyAllWindows()
    
    plt.close()
    print(file)
    image=np.loadtxt(file)
    print(image.shape)
    thresh=temperatureThresholder(image.copy())
    contours,edged=contourMaker(thresh.copy())
    
    filename=os.path.basename(file)
    filename=os.path.splitext(filename)[0]
    
    plt.imsave('mcs_images/'+filename+'_thresh.png',thresh)
    plt.imsave('mcs_images/'+filename+'_image_test.png',image)
    
    plt.imsave('mcs_images/'+filename+'_edged.png',edged)
    mask = np.ones(thresh.shape[:2], dtype="uint8") * 255
    contour_image=cv2.drawContours(image.copy(),contours.copy(),-1,(0,255,0),thickness=15)
    contours_culled=areaCuller(contours.copy(),filename)
    contours_culled=np.array(contours_culled)
    np.save(''.join(['contours_culled/',filename,'_contours.npy']),contours_culled)
    contour_image_culled=cv2.drawContours(image.copy(),contours_culled.copy(),-1,(0,255,0),thickness=15)
    #cv2.imwrite(filename+'contours.png',contour_image)
    #cv2.imwrite(filema,e+'contours_culled.png',contour_image_culled)
    
    #mapProjector(contour_image,'all')
    mapProjector(contour_image_culled,'culled',filename)


print("Done!")


