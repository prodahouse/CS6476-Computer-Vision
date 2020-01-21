# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:00:11 2019

@author: songl
"""
import matplotlib.pyplot as plt
from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists

origImg=plt.imread('fish.jpg')
RGBk5,meanRGBk5=quantizeRGB(origImg,5)
plt.imshow(RGBk5)
plt.title("RGB with k=5")
plt.show()
errorRGBk5=computeQuantizationError(origImg,RGBk5)
HSVk5,meanHSVk5=quantizeHSV(origImg,5)
plt.imshow(HSVk5)
plt.title("HSV with k=5")
plt.show()
errorHSVk5=computeQuantizationError(origImg,HSVk5)
getHueHists(origImg,5)

RGBk25,meanRGBk25=quantizeRGB(origImg,25)
plt.imshow(RGBk25)
plt.title("RGB with k=25")
plt.show()
errorRGBk25=computeQuantizationError(origImg,RGBk25)
HSVk25,meanHSVk25=quantizeHSV(origImg,25)
plt.imshow(HSVk25)
plt.title("HSV with k=25")
plt.show()
errorHSVk25=computeQuantizationError(origImg,HSVk25)
getHueHists(origImg,25)