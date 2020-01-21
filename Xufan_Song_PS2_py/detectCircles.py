# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:54:23 2019

@author: songl
"""

from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import sobel_h, sobel_v
import numpy as np
import matplotlib.pyplot as plt

def detectCircles(im, radius, useGradient):
    """
    Hough Transform circle detector
    
    Args:
        im: input image
        radius: the size of circle we are looking for
        useGradient: flag to exploit gradient orientation
        
    Returns:
        centers: NX2 matrix of detected centers
    """
    # i is row # in y direction, j is col # in x direction
    gray=rgb2gray(im)
    edge=canny(gray,sigma=1).astype(np.int)
    row=np.size(edge,0)
    col=np.size(edge,1)
    gy, gx=np.gradient(edge)
    gy=-gy
    H=np.zeros((row+2*radius,col+2*radius))
    centers=[]
    for i in range(row):
        for j in range(col):
            if edge[i,j]==1:
                if (useGradient==0):
                    for theta in range(360):
                        a=np.around(radius+i+radius*np.sin(np.radians(theta))).astype(np.int)
                        b=np.around(radius+j-radius*np.cos(np.radians(theta))).astype(np.int)
                        H[a,b]+=1;
                else:
                    dx=gx[i,j]
                    dy=gy[i,j]
                    dt=np.around(np.arctan2(dy,dx)).astype(np.int)
                    for theta in range(dt-90,dt+90):
                        a=10*np.around((radius+i+radius*np.sin(np.radians(theta)))/10).astype(np.int)
                        b=10*np.around((radius+j-radius*np.cos(np.radians(theta)))/10).astype(np.int)
                        H[a,b]+=1;
                    
    maximum=np.amax(H)
    plt.imshow(H)
    plt.title("egg accumulator array with binsize=10 r="+str(radius)+" useGradient="+str(useGradient))
    plt.show()
    for a in range(row+2*radius):
        for b in range(col+2*radius):
            if H[a,b]>=maximum*0.9:
                centers+=[[a-radius,b-radius]]
    centers=np.asarray(centers)
    return centers

                  
                    
                
                
    