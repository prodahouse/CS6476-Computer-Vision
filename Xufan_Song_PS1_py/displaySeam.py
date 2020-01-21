# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 22:18:41 2019

@author: songl
"""
import numpy as np
import matplotlib.pyplot as plt
def displaySeam(im, seam, direction):
    """
    display the selected type of seam on top of an imag
    
    Args:
        im: an image of type jpg
        direction: the strings ‘HORIZONTAL’ or ‘VERTICAL’
        
    The output shoulddisplay the input image and plot the seam on top of it
    """
    row=np.size(im,0)
    col=np.size(im,1)
    plt.imshow(im)
    if direction=='HORIZONTAL':
        plt.plot(np.arange(col),seam,'-','r')
    else:
        plt.plot(seam,np.arange(row),'-','r')
    plt.show()
        
        
        
        