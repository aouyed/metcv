#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:07:55 2019

@author: amirouyed
"""

from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from natsort import natsorted 
import cv2
import os
import glob
import numpy as np


ct=CentroidTracker()

rectangles=natsorted(glob.glob('contours_culled/*rectangles.npy'))
images=natsorted(glob.glob('image_arrays/11_*'))

for i,image in enumerate(images):
    rects=[]
    print(image)
    frame=np.loadtxt(image)
    boxes=np.load(rectangles[i])
    for box in boxes:
        (startX, startY, endX, endY) = box.astype("int")
        endX=endX+startX
        endY=endY+startY
        cv2.rectangle(frame, (startX, startY), (endX, endY),
    				(0, 255, 0), 2)
        rects.append((startX, startY, endX, endY))
        objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
    		# draw both the ID of the object and the centroid of the
    		# object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 20, (0, 255, 0), -1)
    filename=os.path.basename(image)
    filename=os.path.splitext(filename)[0]
    cv2.imwrite('tracked_mcs/'+filename+'.png',frame)