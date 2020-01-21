import scipy.io
import matplotlib.pyplot as plt
import pylab as pl
from selectRegion import roipoly
from displaySIFTPatches import displaySIFTPatches
import dist2
import numpy as np

mat = scipy.io.loadmat("twoFrameData.mat")
image1 = mat['im1']
image2 = mat['im2']

pl.imshow(image1)
myRoi = roipoly(roicolor='r')
Index1 = myRoi.getIdx(image1, mat['positions1'])

descriptors1 = mat['descriptors1'][Index1, :]
descriptors2 = mat['descriptors2']
dists = dist2.dist2(descriptors1, descriptors2)
Index2 = np.argmin(dists, axis=1)

fig=plt.figure()
bx=fig.add_subplot(111)
bx.imshow(image2)
corners = displaySIFTPatches(mat['positions2'][Index2,:], mat['scales2'][Index2,:], mat['orients2'][Index2,:])

for j in range(len(corners)):
    bx.plot([corners[j][0][1], corners[j][1][1]], [corners[j][0][0], corners[j][1][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([corners[j][1][1], corners[j][2][1]], [corners[j][1][0], corners[j][2][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([corners[j][2][1], corners[j][3][1]], [corners[j][2][0], corners[j][3][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([corners[j][3][1], corners[j][0][1]], [corners[j][3][0], corners[j][0][0]], color='g', linestyle='-', linewidth=1)
bx.set_xlim(0, image2.shape[1])
bx.set_ylim(0, image2.shape[0])
plt.gca().invert_yaxis()
plt.show()
