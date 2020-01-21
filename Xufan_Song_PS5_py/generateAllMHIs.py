"""
Created on Thu Nov 28 18:11:16 2019

@author: songl
"""

from computeMHI import computeMHI
import matplotlib.pyplot as plt
import numpy as np
import os
MHI = computeMHI('./PS5_Data/botharms/botharms-up-p1-1')
plt.title('Both Arms')
plt.imshow(MHI)
plt.show()
plt.imsave('botharms-up-p1-1_MHI.png',MHI)
np.save('botharms-up-p1-1_MHI.npy', MHI)

MHI = computeMHI('./PS5_Data/crouch/crouch-p1-1')
plt.title('Crouch')
plt.imshow(MHI)
plt.show()
plt.imsave('crouch-p1-1_MHI.png',MHI)
np.save('crouch-p1-1_MHI.npy', MHI)

MHI = computeMHI('./PS5_Data/leftarmup/leftarm-up-p1-1')
plt.title('Left Arm Up')
plt.imshow(MHI)
plt.show()
plt.imsave('leftarm-up-p1-1_MHI.png',MHI)
np.save('leftarm-up-p1-1_MHI.npy', MHI)

MHI = computeMHI('./PS5_Data/punch/punch-p1-1')
plt.title('Punch')
plt.imshow(MHI)
plt.show()
plt.imsave('punch-p1-1_MHI.png',MHI)
np.save('punch-p1-1_MHI.npy', MHI)

MHI = computeMHI('./PS5_Data/rightkick/rightkick-p1-1')
plt.title('Right Kick')
plt.imshow(MHI)
plt.show()
plt.imsave('rightkick-p1-1_MHI.png',MHI)
np.save('rightkick-p1-1_MHI.npy', MHI)

base_directory = './PS5_Data/'
actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
directoryNames = []
MHI = np.zeros((480, 640, 20))
for action in actions:
    directory_name = base_directory + action + '/'
    directoryNames = directoryNames + [directory_name + subdirectory_name for subdirectory_name in os.listdir(directory_name)]

print("length: " + str(len(directoryNames)))
for ith_directory in range(len(directoryNames)):
    temp = directoryNames[ith_directory]
    print("name: " + str(ith_directory) + " " + temp)
    if ".DS_Store" in temp:
      del directoryNames[ith_directory]
      continue
    temp = computeMHI(temp)
    MHI[:,:,ith_directory] = temp
np.save('allMHIs.npy',MHI)
