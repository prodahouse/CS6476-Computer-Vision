# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:49:18 2019

@author: songl
"""
import numpy as np

def computeH(t1,t2):
    """
    take a set of corresponding image pointst1,t2
    both t1 and t2 should be 2xN matrices
    computes the associated 3 x 3 homography matrix H
    """
    row=np.size(t1,0)
    t1=np.append(t1,np.ones((row,1)),axis=1)
    L=np.zeros((row*2,9))
    for i in range(row):
        L[2*i:2*i+1,:]=np.concatenate((t1[i:i+1,:],np.zeros((1,3)),-t2[i,0]*t1[i:i+1,:]),axis=1)
        L[2*i+1:2*i+2,:]=np.concatenate((np.zeros((1,3)),t1[i:i+1,:],-t2[i,1]*t1[i:i+1,:]),axis=1)
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(L.T, L))
    return np.reshape(eigenvectors[:, np.argmin(eigenvalues)],(3, 3))