# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 22:11:31 2019

@author: songl
"""

import numpy as np
def find_optimal_horizontal_seam(cumulativeEnergyMap):
    """
    compute the optimal horizontal seam
    Args:
        cumulativeEnergyMap: 2D matrix of datatypedouble
    Returns:
        horizontalSeam: a vector containing the row indices of the pixels which form the seam for each col.
    """
    row=np.size(cumulativeEnergyMap,0)
    col=np.size(cumulativeEnergyMap,1)
    horizontalSeam=np.zeros(col,dtype=np.int)
    horizontalSeam[col-1]=np.argmin(cumulativeEnergyMap[:,col-1])
    # you use array slicing, but the index returned is the index of the new array
    for i in range(col-1):
        index=horizontalSeam[col-1-i]
        if index==0:
            horizontalSeam[col-2-i]=np.argmin(cumulativeEnergyMap[index:index+2,col-2-i])+index
        elif index==row-1:
            horizontalSeam[col-2-i]=np.argmin(cumulativeEnergyMap[index-1:index+1,col-2-i])+index-1
        else:
            horizontalSeam[col-2-i]=np.argmin(cumulativeEnergyMap[index-1:index+2,col-2-i])+index-1
    return horizontalSeam