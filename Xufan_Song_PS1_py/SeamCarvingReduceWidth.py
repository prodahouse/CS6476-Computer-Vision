# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:37:06 2019

@author: songl
"""

import matplotlib.pyplot as plt
from energy_image import energy_image
from reduceWidth import reduceWidth

pic1=plt.imread('inputSeamCarvingPrague.jpg')
energy1=energy_image(pic1)
for i in range(100):
    pic1, energy1=reduceWidth(pic1, energy1)
plt.imsave('outputReduceWidthPrague.png',pic1)


pic2=plt.imread('inputSeamCarvingMall.jpg')
energy2=energy_image(pic2)
for i in range(100):
    pic2, energy2=reduceWidth(pic2, energy2)
plt.imsave('outputReduceWidthMall.png',pic2)

    
