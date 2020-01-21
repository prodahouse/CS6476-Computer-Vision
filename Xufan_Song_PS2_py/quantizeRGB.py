# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:05:38 2019

@author: songl
"""

from sklearn.cluster import KMeans
import numpy as np

def quantizeRGB(origImg, k):
    """
    quantize the 3-dimensional RGB space, and map each pixel inthe input image to
    its nearest k-means cente
    
    Args:
        origImg: MxNx3matrices of type uint8
        k: specifies the number of colorsto quantize to
    
    Returns:
        outputImg: MxNx3matrices of type uint8
        meanColors: k x 3 array of thekcenters
        [outputImg, meanColors]
    """
    row=np.size(origImg,0)
    col=np.size(origImg,1)  
    img=np.reshape(origImg,(-1,3))
    # convert (M,N,3) to (MXN,3) -1 means inferred from the original array
    mutableImg=np.copy(img) # img from plt.imread is readable only
    kmeans=KMeans(n_clusters=k,random_state=0).fit(mutableImg) 
    label=kmeans.labels_
    meanColors=kmeans.cluster_centers_
    for i in range(row*col):
        mutableImg[i]=meanColors[label[i]]
    outputImg=np.reshape(mutableImg,(row,col,3)).astype(np.uint8)
    return outputImg, meanColors
        