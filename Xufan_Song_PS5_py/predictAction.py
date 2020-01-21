import numpy as np

def predictAction(testMoments, trainMoments, trainLabels):
    distance = np.zeros((1, trainMoments.shape[0]))
    for i in range(0, trainMoments.shape[0]):
        distance[:,i] = np.sqrt(np.sum(np.divide(np.power(np.reshape(trainMoments[i,:], (-1, 1)) -
            np.reshape(testMoments,(-1, 1)), 2), np.reshape(np.nanvar(trainMoments, axis = 0), (-1, 1)))))
    return int(trainLabels[np.argsort(distance)][0, 0])
