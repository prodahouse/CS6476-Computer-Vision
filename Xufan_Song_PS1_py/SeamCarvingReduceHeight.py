# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 01:17:38 2019

@author: songl
"""

import matplotlib.pyplot as plt
from energy_image import energy_image
from reduceHeight import reduceHeight

pic1=plt.imread('inputSeamCarvingPrague.jpg')
energy1=energy_image(pic1)
for i in range(100):
    pic1, energy1=reduceHeight(pic1, energy1)
plt.imsave('outputReduceHeightPrague.png',pic1)


pic2=plt.imread('inputSeamCarvingMall.jpg')
energy2=energy_image(pic2)
for i in range(100):
    pic2, energy2=reduceHeight(pic2, energy2)
plt.imsave('outputReduceHeightMall.png',pic2)