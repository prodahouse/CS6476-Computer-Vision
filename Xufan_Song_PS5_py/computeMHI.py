"""
Created on Thu Nov 28 18:11:16 2019

@author: songl
"""


import glob
import numpy as np
from imageio import imread


def computeMHI(directoryName):
    directory = np.sort(glob.glob(directoryName + '/' '*.pgm'))
    leng = len(directory)
    MHI = np.zeros((imread(directory[0]).shape[0], imread(directory[0]).shape[1]))
    for i in range(leng):
        depth = imread(directory[i])
        depth[depth > 39000] = 0
        depth[depth != 0] = 1
        if i > 0:
            diff = np.absolute(depth - prev)
            MHI[diff == 1] = leng
            MHI[diff != 1] = MHI[diff != 1] - 1
            MHI[MHI < 0] = 0
            prev = depth
        else:
            prev = depth
    MHI = MHI/np.amax(MHI)
    return MHI
