import glob
import pickle
import scipy.io
import pylab as pl
from scipy import misc
from selectRegion import roipoly
import numpy as np
import heapq
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

def computeSimilar(freq1, freq2):
    return np.sum(freq1 * freq2) / (np.linalg.norm(freq1) * np.linalg.norm(freq2))

siftdir = 'sift/'
framesdir = 'frames/'
filenames = glob.glob(siftdir + '*.mat')
num_frames = len(filenames)

frequency = np.zeros([num_frames, 1500])
descriptors = scipy.io.loadmat(filenames[0])['descriptors']
for i in range(1, len(filenames)):
    mat = scipy.io.loadmat(filenames[i], verify_compressed_data_integrity=False)
    descriptor = mat['descriptors']
    descriptors = np.vstack((descriptors, descriptor))
kmeans_model = MiniBatchKMeans(n_clusters=1500, batch_size=100000).fit(descriptors)

for i in range(len(filenames)):
    mat = scipy.io.loadmat(filenames[i], verify_compressed_data_integrity=False)
    descriptors = mat['descriptors']
    if (len(descriptors)) == 0:
        continue
    clusters = kmeans_model.predict(descriptors)
    for cluster in clusters:
        frequency[i][cluster] += 1

#calculate the frequency of chosen region
mat = scipy.io.loadmat(filenames[341], verify_compressed_data_integrity=False)
imagename = framesdir + mat['imagename'][0]
im = misc.imread(imagename)

for k in range(4):
    freq = np.zeros(1500)
    pl.imshow(im)
    myRoi = roipoly(roicolor='r')
    descriptors = mat['descriptors']
    clusters = kmeans_model.predict(descriptors)
    for cluster in clusters:
        freq[cluster] += 1
    score = []
    for i in range(len(filenames)):
        if np.max(frequency[i]) == 0:
            continue
        score.append((computeSimilar(freq, frequency[i]), i))
    heapq.heapify(score)
    results = heapq.nlargest(6, score)
    for result in results:
        index = result[1]
        score = result[0]
        mat = scipy.io.loadmat(filenames[index], verify_compressed_data_integrity=False)
        imagename = framesdir + mat['imagename'][0]
        image = plt.imread(imagename)
        plt.imsave(str(k + 5) + 'frame' + str(index)+ 'score' + str(score) + '.png', image)
