# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 19:32:07 2019

@author: songl
"""
import cv2
import numpy as np
def energy_image(im):
    """
    compute the energy at each pixel using the magnitude of the x and y 
    gradients,equation 1 in the paper
    Args:
        im:MxNx3matrix of datatype uint8       

    Returns:
        energyImage: 2D matrix of dataype double
    """
    im_gray=np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])
    # convert color to grayscale or we can try to compute for each channel
    sobelx=cv2.Sobel(im_gray,cv2.CV_64F,1,0,ksize=3)
    sobely=cv2.Sobel(im_gray,cv2.CV_64F,0,1,ksize=3)
    energyImage=np.absolute(sobelx)+np.absolute(sobely)
    return energyImage