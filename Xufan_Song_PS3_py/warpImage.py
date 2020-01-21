# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:06:14 2019

@author: songl
"""
import numpy as np
from numpy.linalg import inv

def warpImage(inputIm, refIm, H):
    TotalRow=np.size(inputIm,0)
    TotalCol=np.size(inputIm,1)
    n = np.dot(H, np.array([[0, TotalCol - 1, TotalCol - 1, 0],
        [0 , 0, TotalRow - 1, TotalRow - 1],
        [1, 1, 1, 1]]))
    temp = n[2, :]
    t = np.zeros((np.size(n,0) - 1, np.size(n,1)))
    t[0, :] = np.around(n[0, :] / temp)
    t[1, :] = np.around(n[1, :] / temp)
    p, q = np.meshgrid(np.arange(min(np.amin(t[0,:]),0), np.amax(t[0,:]), 1),
        np.arange(min(np.amin(t[1,:]),0), np.amax(t[1,:]), 1))
    nc = np.dot(inv(H), np.array([np.ravel(p),np.ravel(q),np.ravel(np.ones(p.shape))]))
    c = np.zeros((np.size(nc,0) - 1, np.size(nc,1)))
    c[0,:] = np.around(nc[0,:] / nc[2,:])
    c[1,:] = np.around(nc[1,:] / nc[2,:])
    c = np.reshape(np.transpose(c),(np.size(p,0),np.size(p,1),2))
    warpIm = np.zeros((np.size(p,0),np.size(p,1),3))
    for i in range(0, np.size(warpIm,0)):
        for j in range(0,np.size(warpIm,1)):
            if c[i, j, 0] < 0 or c[i, j, 0] >  TotalCol - 1 or c[i, j, 1] < 0 or c[i, j, 1] > TotalRow - 1: warpIm[i, j] = np.array([0, 0, 0])
            else: warpIm[i, j] = np.array(inputIm[int(c[i, j, 1]), int(c[i, j, 0]),:])
    mergeIm = np.copy(warpIm)
    mergeIm = np.asarray(mergeIm, dtype = np.uint8)
    mergeIm[abs(int(min(np.amin(t[1,:]),0))): abs(int(min(np.amin(t[1,:]),0))) + np.size(refIm,0), abs(int(min(np.amin(t[0, :]),0))):abs(int(min(np.amin(t[0, :]),0))) + np.size(refIm,1), :] = refIm
    return np.asarray(warpIm, dtype=np.uint8), mergeIm