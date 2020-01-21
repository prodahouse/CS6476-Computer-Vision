import glob
import pickle
import numpy as np
import scipy.io
import heapq
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

def computeSimilar(freq1, freq2):
    return np.sum(freq1 * freq2) / (np.linalg.norm(freq1) * np.linalg.norm(freq2))

siftdir = 'sift/'
framesdir = 'frames/'
filenames = glob.glob(siftdir + '*.mat')
num_frames = len(filenames)

descriptors = scipy.io.loadmat(filenames[0])['descriptors']
for i in range(1, len(filenames)):
    mat = scipy.io.loadmat(filenames[i], verify_compressed_data_integrity=False)
    descriptor = mat['descriptors']
    descriptors = np.vstack((descriptors, descriptor))
kmeans_model = MiniBatchKMeans(n_clusters=1500, batch_size=100000).fit(descriptors)

frequency = np.zeros([num_frames, 1500])

for i in range(len(filenames)):
    mat = scipy.io.loadmat(filenames[i], verify_compressed_data_integrity=False)
    descriptors = mat['descriptors']
    if (len(descriptors)) == 0:
        continue
    clusters = kmeans_model.predict(descriptors)
    for cluster in clusters:
        frequency[i][cluster] += 1

frame1 = 1
frame2 = 3
frame3 = 5

score1 = []
score2 = []
score3 = []

for i in range(len(filenames)):
    if np.max(frequency[i]) == 0:
        continue
    score1.append((computeSimilar(frequency[frame1], frequency[i]), i))
    score2.append((computeSimilar(frequency[frame2], frequency[i]), i))
    score3.append((computeSimilar(frequency[frame3], frequency[i]), i))

heapq.heapify(score1)
heapq.heapify(score2)
heapq.heapify(score3)

results1 = heapq.nlargest(6, score1)
results2 = heapq.nlargest(6, score2)
results3 = heapq.nlargest(6, score3)

for result in results1:
    index = result[1]
    score = result[0]
    mat = scipy.io.loadmat(filenames[index], verify_compressed_data_integrity=False)
    imagename = framesdir + mat['imagename'][0]
    im = plt.imread(imagename)
    plt.imsave('frame1' + str(index)+ 'score' + str(score) + '.png', im)

for result in results2:
    index = result[1]
    score = result[0]
    mat = scipy.io.loadmat(filenames[index], verify_compressed_data_integrity=False)
    imagename = framesdir + mat['imagename'][0]
    im = plt.imread(imagename)
    plt.imsave('frame2' + str(index)+ 'score' + str(score) + '.png', im)

for result in results3:
    index = result[1]
    score = result[0]
    mat = scipy.io.loadmat(filenames[index], verify_compressed_data_integrity=False)
    imagename = framesdir + mat['imagename'][0]
    im = plt.imread(imagename)
    plt.imsave('frame3' + str(index)+ 'score' + str(score) + '.png', im)
