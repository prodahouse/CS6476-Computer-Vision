# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:46:16 2019

@author: songl
"""

import numpy as np
def cumulative_minimum_energy_map(energyImage, seamDirection):
    """
    compute minimum cumulative energy
    Args:
        energyImage: a 2D matrix of datatype double.(It can be the output of
                     energy_image function defined above
        seamDirection: strings ‘HORIZONTAL’ or ‘VERTICAL’
    Returns:
        cumulativeEnergyMap: 2D matrix of datatypedouble
    """
    row=np.size(energyImage,0)
    col=np.size(energyImage,1)
    cumulativeEnergyMap=np.zeros((row,col))
    if seamDirection=='HORIZONTAL':
        cumulativeEnergyMap[:,0:1]=energyImage[:,0:1]
        for j in range(1,col):
            for i in range(row):
                if i==0:
                    cumulativeEnergyMap[i,j]=min(cumulativeEnergyMap[i,j-1],cumulativeEnergyMap[i+1,j-1])+energyImage[i,j]
                elif i==row-1:
                    cumulativeEnergyMap[i,j]=min(cumulativeEnergyMap[i,j-1],cumulativeEnergyMap[i-1,j-1])+energyImage[i,j]
                else:
                    cumulativeEnergyMap[i,j]=min(cumulativeEnergyMap[i-1,j-1],cumulativeEnergyMap[i,j-1],cumulativeEnergyMap[i+1,j-1])+energyImage[i,j]
    elif seamDirection=='VERTICAL':
        cumulativeEnergyMap[0:1,:]=energyImage[0:1,:]
        for i in range(1,row):
            for j in range(col):
                if j==0:
                    cumulativeEnergyMap[i,j]=min(cumulativeEnergyMap[i-1,j],cumulativeEnergyMap[i-1,j+1])+energyImage[i,j]
                elif j==col-1:
                    cumulativeEnergyMap[i,j]=min(cumulativeEnergyMap[i-1,j],cumulativeEnergyMap[i-1,j-1])+energyImage[i,j]
                else:
                    cumulativeEnergyMap[i,j]=min(cumulativeEnergyMap[i-1,j-1],cumulativeEnergyMap[i-1,j],cumulativeEnergyMap[i-1,j+1])+energyImage[i,j]
    return cumulativeEnergyMap
                   
                    
        

        
        
