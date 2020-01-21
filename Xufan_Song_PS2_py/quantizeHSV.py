# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:52:56 2019

@author: songl
"""
from sklearn.cluster import KMeans
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

def quantizeHSV(origImg, k):
    """
    Given an RGB image, convert to HSV, and quantize the 1-dimensional Hue space
    while keeping its Saturation and Value channels the same as the input
    
    Args:
        origImg: MxNx3matrices of type uint8
        k: specifies the number of colorsto quantize to
    
    Returns:
        outputImg: MxNx3matrices of type uint8
        meanHues: k x 1vector of the hue center
    """
    row=np.size(origImg,0)
    col=np.size(origImg,1)
    hsvImg=rgb2hsv(origImg)
    mutableHsvImg=np.copy(hsvImg)
    imgToDo=np.reshape(mutableHsvImg[:,:,0],(-1,1))
    kmeans=KMeans(n_clusters=k,random_state=0).fit(imgToDo) 
    label=kmeans.labels_
    meanHues=kmeans.cluster_centers_
    for i in range(row*col):
        imgToDo[i]=meanHues[label[i]]
    mutableHsvImg[:,:,0]=np.reshape(imgToDo,(row,col))
    outputImg=hsv2rgb(mutableHsvImg)
    outputImg=(255*outputImg/np.amax(outputImg)).astype(np.uint8)
    return outputImg, meanHues