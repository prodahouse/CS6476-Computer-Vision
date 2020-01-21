# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:53:14 2019

@author: songl
"""
import numpy as np
def find_optimal_vertical_seam(cumulativeEnergyMap):
    """
    compute the optimalvertical seam
    Args:
        cumulativeEnergyMap: 2D matrix of datatypedouble
    Returns:
        verticalSeam: a vector containing the column indices of the pixels which form the seam for each row.
    """
    row=np.size(cumulativeEnergyMap,0)
    col=np.size(cumulativeEnergyMap,1)
    verticalSeam=np.zeros(row,dtype=np.int)
    verticalSeam[row-1]=np.argmin(cumulativeEnergyMap[row-1,:])
    for i in range(row-1):
        index=verticalSeam[row-1-i]
        if index==0:
            verticalSeam[row-2-i]=np.argmin(cumulativeEnergyMap[row-2-i,index:index+2])+index
        elif index==col-1:
            verticalSeam[row-2-i]=np.argmin(cumulativeEnergyMap[row-2-i,index-1:index+1])+index-1
        else:
            verticalSeam[row-2-i]=np.argmin(cumulativeEnergyMap[row-2-i,index-1:index+2])+index-1
    return verticalSeam
    