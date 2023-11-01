import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips
import persim
from persim import PersistenceImager
import loader
import masker

'''
TODO:  1 - Finish at least one data extraction technique (in getData)
            a. Should flatten the data (if needed) before returning a copy of it
            b. Should return both the data and labels, separately for the sake of train_test_split
       2 - Add persistence landscape and other calculations to the object
       3 - Add an option to save or load a pre-made TopologicalfMRI object
       4 - Setup code to run all of the files from one subject so we can start running tests 
'''


#*************************** CREATE RIPS DIAGRAMS ***************************#


class TopologicalfMRI:
    '''
    Input:
    Purpose: Hold the rips diagrams made from a set of training data and calculate/returns the topological features
     needed for training and SVM or linear model using TDA.
    '''
    def __init__(self, data, labels, mask, nTime=3, ripsMaxDim=1, ripsCoeff=2):
        self._mask = mask.copy()
        self._labels = labels
        self._ripsMaxDim = ripsMaxDim
        self._ripsCoeff = ripsCoeff
        self._pimgr = None
        # no need to store the data we only need rips after
        flatData = self._trimFlatten(data, self._mask, nTime)
        self._rips = Rips(maxdim=self._ripsMaxDim, coeff=self._ripsCoeff)
        self._topologicalFeatures = {'diagrams': [self._rips.fit_transform(flatData[:, :, i])
                                                  for i in range(flatData.shape[2])]}

    def _trimFlatten(self, data, mask, nTime):
        # return the dataset flattened without the mask
        d1, d2, d3, tPoints = data.shape
        # flatten and trim data based on mask, expect 3 data points and trimmed rest periods
        flatData = np.empty((int(np.sum(mask)), nTime, int(tPoints / nTime)))
        keepDim = np.nonzero(mask)

        for i in range(0, tPoints, nTime):
            for j in range(0, nTime):
                flatData[:, j, int(i / nTime)] = np.reshape(data[keepDim][:, i + j], -1)

        return flatData

    def addData(self, data, labels, nTime):
        flatData = self._trimFlatten(data, self._mask, nTime)
        self._topologicalFeatures['diagrams'] += [self._rips.fit_transform(flatData[:, :, i])
                                                  for i in range(flatData.shape[2])]
        self._labels += labels

    def calculatePersistenceImages(self, data=None, nTime=None):
        if data is None or self._pimgr is None:
            self._pimgr = PersistenceImager(pixel_size=1)
            # keep in form that needs to be flattened when used
            topFeatLen = len(self._topologicalFeatures['diagrams'])
            self._topologicalFeatures['PI'] = self._pimgr.fit_transform([self._topologicalFeatures['diagrams'][i][1]
                                                                         for i in range(topFeatLen)])
        if data is not None:
            flatData = self._trimFlatten(data, self._mask, nTime)
            return self._pimgr.transform([self._rips.fit_transform(flatData[:, i])[1]
                                          for i in range(flatData.shape[2])])

    def getData(self, dataType='PI'):
        # check dataType and only return those for prediction
        if dataType == 'PI':
            return self._topologicalFeatures['PI'].copy().flatten()
        else:
            return self._topologicalFeatures.copy()
