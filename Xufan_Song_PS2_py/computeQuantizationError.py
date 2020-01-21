# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:53:18 2019

@author: songl
"""
import numpy as np
def computeQuantizationError(origImg, quantizedImg):
    """
    compute the SSD error (sum of squared error) between the original RGB pixel values
    and the quantized value
    """
    error=np.sum(np.power(origImg-quantizedImg,2))
    return error