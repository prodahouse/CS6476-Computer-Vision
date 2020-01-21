# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:14:38 2019

@author: songl
"""

import numpy as np
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_vertical_seam import find_optimal_vertical_seam
def reduceWidth(im, energyImage):
    """
    Args:
        energyImage: 2D matrix of datatype double
        im: MxNx3 matrix of datatype uint8
        
    Returns:
        reducedColorImage: 3D matrix same as the input image but with its width reduced by one pixel
        reducedEnergyImage: 2D matrix same as the inputenergyImage, but with its width reduced by one pixel
    """
    row=np.size(energyImage,0)
    col=np.size(energyImage,1)
    cumap=cumulative_minimum_energy_map(energyImage, 'VERTICAL')
    seam=find_optimal_vertical_seam(cumap)
    reducedColorImage=np.zeros((row,col-1,3),dtype=np.uint8)
    reducedEnergyImage=np.zeros((row,col-1),dtype=np.double)
    for i in range(row):
        reducedColorImage[i,0:seam[i],:]=im[i,0:seam[i],:]
        reducedEnergyImage[i,0:seam[i]]=energyImage[i,0:seam[i]]
        reducedColorImage[i,seam[i]:col-1,:]=im[i,seam[i]+1:col,:]
        reducedEnergyImage[i,seam[i]:col-1]=energyImage[i,seam[i]+1:col]
    return reducedColorImage, reducedEnergyImage
    