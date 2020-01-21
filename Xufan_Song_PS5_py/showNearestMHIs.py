import os
import numpy as np
import matplotlib.pyplot as plt
from computeMHI import computeMHI
from huMoments import huMoments

def showNearestMHIs(testMoments, trainMoments, trainDirectoryNames, K):
    distance = np.zeros((1, trainMoments.shape[0]))
    nMHI = np.zeros((480, 640, K+1))
    for i in range(0, trainMoments.shape[0]):
        distance[:,i] = np.sqrt(np.sum(np.divide(np.power(np.reshape(trainMoments[i,:],(-1,1)) -
            np.reshape(testMoments,(-1,1)),2), np.reshape(np.nanvar(trainMoments,axis=0),(-1,1)))))
    sorted_train_directory_names = np.reshape(trainDirectoryNames[np.argsort(distance)],(trainDirectoryNames.shape))
    for i in range(0, K+1):
        nMHI[:,:,i] = computeMHI(sorted_train_directory_names[i,:][0])
    return nMHI

if __name__ == "__main__":
    actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
    trainDirectoryNames = []
    trainMoments = np.asarray(np.load('huVectors.npy'))
    K = 4
    for action in actions:
        directory_name = './PS5_Data/' + action + '/'
        trainDirectoryNames = trainDirectoryNames + [directory_name + subdirectory_name for subdirectory_name in os.listdir(directory_name)]
    trainDirectoryNames = np.reshape(trainDirectoryNames,(-1,1))

    nMHI = showNearestMHIs(huMoments(np.load('botharms-up-p1-1_MHI.npy')), trainMoments, trainDirectoryNames, K)
    plt.title("Chosen MHI - botharms-up-p1-1_MHI")
    plt.imshow(nMHI[:, :, 0])
    plt.show()
    for ith_MHI in range(1,K + 1):
        plt.title("Nearest Neighbor " + str(ith_MHI) +" to botharms-up-p1-1 MHI")
        plt.imshow(nMHI[:,:,ith_MHI])
        plt.show()

    nMHI = showNearestMHIs(huMoments(np.load('crouch-p1-1_MHI.npy')), trainMoments, trainDirectoryNames, K)
    plt.title("Chosen MHI - crouch-p1-1_MHI")
    plt.imshow(nMHI[:, :, 0])
    plt.show()
    for ith_MHI in range(1,K + 1):
        plt.title("Nearest Neighbor " + str(ith_MHI) +" to crouch-p1-1 MHI")
        plt.imshow(nMHI[:,:,ith_MHI])
        plt.show()

    nMHI = showNearestMHIs(huMoments(np.load('leftarm-up-p1-1_MHI.npy')), trainMoments, trainDirectoryNames, K)
    plt.title("Chosen MHI - leftarm-up-p1-1_MHI")
    plt.imshow(nMHI[:, :, 0])
    plt.show()
    for ith_MHI in range(1,K + 1):
        plt.title("Nearest Neighbor " + str(ith_MHI) +" to leftarm-up-p1-1 MHI")
        plt.imshow(nMHI[:,:,ith_MHI])
        plt.show()

    nMHI = showNearestMHIs(huMoments(np.load('punch-p1-1_MHI.npy')), trainMoments, trainDirectoryNames, K)
    plt.title("Chosen MHI - punch-p1-1_MHI")
    plt.imshow(nMHI[:, :, 0])
    plt.show()
    for ith_MHI in range(1,K + 1):
        plt.title("Nearest Neighbor " + str(ith_MHI) +" to punch-p1-1 MHI")
        plt.imshow(nMHI[:,:,ith_MHI])
        plt.show()

    nMHI = showNearestMHIs(huMoments(np.load('rightkick-p1-1_MHI.npy')), trainMoments, trainDirectoryNames, K)
    plt.title("Chosen MHI - rightkick-p1-1_MHI")
    plt.imshow(nMHI[:, :, 0])
    plt.show()
    for ith_MHI in range(1,K + 1):
        plt.title("Nearest Neighbor " + str(ith_MHI) +" to rightkick-p1-1 MHI")
        plt.imshow(nMHI[:,:,ith_MHI])
        plt.show()
