# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:58:52 2019

@author: songl
"""
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
import numpy as np
from sklearn.cluster import KMeans

def getHueHists(im,k):
    """
    compute and display two histograms of its hue value
    
    Returns:
        histEqual: equally-spaced bins (uniformly dividing up the hue values)
        histClustered: use bins defined by the k cluster center membership
    """
    imHsv=rgb2hsv(im)
    mutableImHsv=np.copy(imHsv)
    imgToDo=np.reshape(mutableImHsv[:,:,0],(-1,1))
    plt.hist(imgToDo,bins=k)
    plt.title("Histogram for histEqual with k="+str(k))
    plt.show()
    kmeans=KMeans(n_clusters=k,random_state=0).fit(imgToDo) 
    label=kmeans.labels_
    meanHues=kmeans.cluster_centers_
    # meanHues is not in the increasing order for corresponding label
    meanHuesSort=np.sort(meanHues,axis=0)
    labelNum=np.zeros(k).astype(np.int)
    length=np.size(label)
    for i in range(length):
        for j in range(k):
            if meanHues[label[i]]==meanHuesSort[j]:
                labelNum[j]+=1
    imgToDoSort=np.sort(imgToDo,axis=0)
    binArr=np.zeros(k+1)
    binArr[k]=imgToDoSort[length-1,0]
    SUM=0
    for i in range(1,k):
        SUM+=labelNum[i-1]
        binArr[i]=imgToDoSort[SUM,0]
    plt.hist(imgToDo,bins=binArr)
    plt.title("Histogram for histClustered with k="+str(k))
    plt.show()
    

    
    
      
    
    