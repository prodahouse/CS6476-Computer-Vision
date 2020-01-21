import scipy.io
import glob
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import misc

siftdir = 'sift/'
framesdir = 'frames/'
filenames = glob.glob(siftdir + '*.mat')

descriptors = scipy.io.loadmat(filenames[0])['descriptors']
for i in range(1, len(filenames)):
    mat = scipy.io.loadmat(filenames[i], verify_compressed_data_integrity=False)
    descriptor = mat['descriptors']
    descriptors = np.vstack((descriptors, descriptor))
kmeans_model = MiniBatchKMeans(n_clusters=1500, batch_size=100000).fit(descriptors)

patches1 = []
patches2 = []

for filename in filenames:
    mat = scipy.io.loadmat(filename, verify_compressed_data_integrity=False)
    descriptors = mat['descriptors']
    if(len(descriptors) == 0):
        continue
    imagename = framesdir + mat['imagename'][0]
    im = misc.imread(imagename)
    clusters = kmeans_model.predict(descriptors)
    for i in range(len(clusters)):
        cluster = clusters[i]
        if cluster == 111:
            image_patch = getPatchFromSIFTParameters(mat['positions'][i,:], mat['scales'][i], mat['orients'][i], rgb2gray(im))
            patches1.append(image_patch)
        elif cluster == 222:
            image_patch = getPatchFromSIFTParameters(mat['positions'][i,:], mat['scales'][i], mat['orients'][i], rgb2gray(im))
            patches2.append(image_patch)
    if len(patches1) >= 25 and len(patches2) >= 25:
        break

fig=plt.figure()
columns = 5
rows = 5
for i in range(1, columns * rows +1):
    image = patches1[i - 1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(image, cmap='gray')
plt.show()

fig=plt.figure()
columns = 5
rows = 5
for i in range(1, columns * rows +1):
    image = patches2[i - 1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(image, cmap='gray')
plt.show()
