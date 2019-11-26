#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:34:16 2019

@author: amirouyed
"""
import numpy as np
import cv2

def flowArrows( Image,Flow,Divisor,scale,name,color,tipLength ):
    #Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
    PictureShape = np.shape(Image)
    #determine number of quiver points there will be
    Imax = int(PictureShape[0]/Divisor)
    Jmax = int(PictureShape[1]/Divisor)
    print(Flow.shape)
    Image=cv2.cvtColor(Image,cv2.COLOR_GRAY2BGR)  

    #create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
      for j in range(1, Jmax):
         X1 = (i)*Divisor
         Y1 = (j)*Divisor
         X2 = int(X1 + scale*Flow[X1,Y1,1])
         Y2 = int(Y1 + scale*Flow[X1,Y1,0])
         #X2 = np.clip(X2, 0, PictureShape[1])
         #Y2 = np.clip(Y2, 0, PictureShape[0])
         #add all the lines to the mask
         mask = cv2.arrowedLine(mask, (Y1,X1),(Y2,X2), color, 3, tipLength=tipLength)
               
    #superpose lines onto image
    img = cv2.add(Image,mask)


    return (img)