# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:35:19 2019

@author: songl
"""

import numpy as np
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
def reduceHeight(im, energyImage):
    """
    Args:
        energyImage: 2D matrix of datatype double
        im: MxNx3 matrix of datatype uint8
        
    Returns:
        reducedColorImage: 3D matrix same as the input image but with its height reduced by one pixel
        reducedEnergyImage: 2D matrix same as the inputenergyImage, but with its height reduced by one pixel
    """
    row=np.size(energyImage,0)
    col=np.size(energyImage,1)
    cumap=cumulative_minimum_energy_map(energyImage, 'HORIZONTAL')
    seam=find_optimal_horizontal_seam(cumap)
    reducedColorImage=np.zeros((row-1,col,3),dtype=np.uint8)
    reducedEnergyImage=np.zeros((row-1,col),dtype=np.double)
    for i in range(col):
        reducedColorImage[0:seam[i],i,:]=im[0:seam[i],i,:]
        reducedEnergyImage[0:seam[i],i]=energyImage[0:seam[i],i]
        reducedColorImage[seam[i]:row-1,i,:]=im[seam[i]+1:row,i,:]
        reducedEnergyImage[seam[i]:row-1,i]=energyImage[seam[i]+1:row,i]
    return reducedColorImage, reducedEnergyImage