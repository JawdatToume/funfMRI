import re

import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips
import persim
from persim import PersistenceImager
import loader
import masker
import pickle
import pandas as pd
import os

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
    def __init__(self, data, labels, mask, nTime=3, ripsMaxDim=1, ripsCoeff=2, keepRange=(0, 2), saveMask=True):
        self._mask = mask.copy() if mask is not None else []
        self._labels = labels if labels is not None else []
        self._ripsMaxDim = ripsMaxDim
        self._ripsCoeff = ripsCoeff
        self._pimgr = None
        self._keepRange = keepRange
        # no need to store the data we only need rips after
        flatData = self._trimFlatten(data, self._mask, nTime) if data is not None else None
        self._rips = Rips(maxdim=self._ripsMaxDim, coeff=self._ripsCoeff) # add back +3 later maybe
        self._topologicalFeatures = {'diagrams': [self._rips.fit_transform(flatData[:, self._keepRange[0]:(self._keepRange[1]), i])[1]
                                                  for i in range(flatData.shape[2])]} if data is not None else \
            {'diagrams': []}
        if not saveMask:
            self._mask = []


    def __add__(self, other):
        newTDA = TopologicalfMRI(None, None, None)
        newTDA._mask = self._mask.copy()
        newTDA._labels = np.append(self._labels.copy(), other._labels.copy())
        newTDA._keepRange = self._keepRange
        newTDA._topologicalFeatures = {'diagrams': self._topologicalFeatures['diagrams'].copy()}
        newTDA._topologicalFeatures['diagrams'] += other._topologicalFeatures['diagrams'].copy()
        return newTDA

    def _trimFlatten(self, data, mask, nTime):
        # return the dataset flattened without the mask
        d1, d2, d3, tPoints = data.shape
        # flatten and trim data based on mask, expect 3 data points and trimmed rest periods
        flatData = np.empty((int(np.sum(mask)), nTime + 3, int(tPoints / nTime)))
        #flatData = np.empty((int(np.sum(mask)), nTime, int(tPoints / nTime)))
        keepDim = np.nonzero(mask)

        for i in range(0, tPoints, nTime):
            for j in range(0, nTime):
                if j < nTime:
                    temp = np.reshape(data[keepDim][:, i + j], -1)
                    #norm_mean = np.mean(temp)
                    #norm_scale = np.std(temp, ddof=1)
                    #flatData[:, j, int(i / nTime)] = (temp - norm_mean) / norm_scale
                    flatData[:, j, int(i / nTime)] = temp
                #else:
                #    temp = np.array(keepDim[j-nTime]).flatten()
                #    norm_mean = np.mean(temp)
                #    norm_scale = np.std(temp, ddof=1)
                #    flatData[:, j, int(i / nTime)] = (temp - norm_mean) / norm_scale

        return flatData

    def addData(self, data, labels, nTime, keepRange=None, mask=None):
        if mask is None:
            mask = self._mask
        assert mask is None, "Mask not saved, set a mask"
        if keepRange is None:
            keepRange = self._keepRange
        flatData = self._trimFlatten(data, mask, nTime)
        self._topologicalFeatures['diagrams'] += [self._rips.fit_transform(flatData[:, keepRange[0]:keepRange[1], i])[1]
                                                  for i in range(flatData.shape[2])]
        self._labels = np.append(self._labels, labels)

    def calculatePersistenceImages(self, pimgr=None):
        if pimgr is None:
            pimgr = self._pimgr
            if self._pimgr is None:
                self._pimgr = PersistenceImager(pixel_size=1)
                self._topologicalFeatures['PI'] = self._pimgr.fit_transform(self._topologicalFeatures['diagrams'])
            return None
        else:
            self._pimgr = pimgr
            self._topologicalFeatures['PI'] = self._pimgr.transform(self._topologicalFeatures['diagrams'])
            return None

    def getData(self, dataType='PI'):
        # check dataType and only return those for prediction
        if dataType == 'PI':
            data = self._topologicalFeatures['PI'].copy()
            new_data = np.empty((len(data), data[0].shape[0]*data[0].shape[1]))
            for i in range(len(data)):
                new_data[i, :] = data[i].flatten()
            return new_data, self._labels
        else:
            return self._topologicalFeatures.copy()

    def setLabels(self, newLabels):
        assert len(newLabels) == len(self._labels), "New and old labels must have same length"
        self._labels = newLabels.copy()


# add a piece of code that goes through everything and saves it (reading event files too)
def convertTDA(fileType="perceptionTraining"):
    if fileType == "perceptionTraining":
        typeNum = 2
    elif fileType == "perceptionTest":
        typeNum = 1
    elif fileType == "imageryTest":
        typeNum = 0
    else:
        raise Exception("fileType specified but not one of perceptionTraining, perceptionTest, or imageryTest.")
    for subj in range(1, 6):
        # get the number of runs
        m1 = masker.maskSubject(subj, "VC")
        fileMatches = [int(f[-2:]) for f in os.listdir("fMRIFullData/sub-0" + str(subj) + "/")
                       if re.match("ses-" + fileType + "*", f)]
        runNum = max(fileMatches)
        for run in range(1, runNum + 1):
            # get the number of trials
            fileMatches = [int(f[-14:-12]) for f in
                           os.listdir("fMRIFullData/sub-0" + str(subj) + "/ses-" + fileType + "%02i"
                                      % run + "/func/")
                           if re.match(
                    "sub-0" + str(subj) +
                    "_ses-" + fileType + "0" + str(run) +
                    "_task-perception_run-0*", f)]
            trialNum = max(fileMatches)
            for trial in range(1, trialNum + 1):
                # set-up a pickle string
                pickleString = ("subject_" + str(subj) + "_" + fileType + "_run_" + str(run) + "_trial_" + str(trial) +
                                "_VC_CORR_SHIFT.pickle")
                print("Running ", pickleString)

                # read and convert the given nii.gz file with mask passing in the events as labels
                # open data
                data = loader.loaderSpecific(subj, typeNum, run, trial)
                # cutoff the first few timepoints that are only rest, same with the last few
                data = data[:, :, :, 11:(data.shape[3] - 2)]
                # fit rips and object with labels for class
                events = pd.read_csv("fMRIFullData/sub-0" + str(subj) +
                                     "/ses-" + fileType + "0" + str(run) +
                                     "/func/sub-0" + str(subj) +
                                     "_ses-" + fileType + "0" + str(run) +
                                     "_task-perception_run-%02i" % trial + "_events.tsv", sep='\t')
                tf = TopologicalfMRI(data, events['stimulus_name'][1:(events['stimulus_name'].shape[0] - 1)],
                                     m1, nTime=3, keepRange=(0, 2),
                                     saveMask=False)

                #tf.setLabels(np.array(events['stimulus_name'][1:(events['stimulus_name'].shape[0] - 1)]))
                print(tf._labels)  # confirm labels are correct

                # write the pickle file in top directory
                with open(pickleString, "wb") as f:
                    pickle.dump(tf, f)


def joinTDA(subject=1, fileType="perceptionTraining", prePI=None, save=True, suffix="CORR_SHIFT_CLUSTERS"):
    # this will open and join all the files with this name use the code in above
    #  to get these
    joinedObj = None
    if type(subject) is not list:
        subject = [subject]
    for sub in subject:
        fileMatches = [f for f in os.listdir(".") if re.match(r"subject_" + str(sub) + "_" + fileType +
                                                              ".*" + suffix + "\.pickle", f)]
        for fileName in fileMatches:
            print(fileName)
            with open(fileName, "rb") as f:
                temp = pickle.load(f)
            #print("Read Labels")
            #print(temp._labels)
            if joinedObj is None:
                joinedObj = temp
            else:
                joinedObj = joinedObj + temp
            #print("Joined Labels")
            #print(joinedObj._labels)
            #print("What it should be:")
            #print(np.append(joinedObj._labels, temp._labels))

    # after that is opened and joined calculate the PI (either use a given
    #  one or calculate within the object)
    joinedObj.calculatePersistenceImages(pimgr=prePI)
    joinedObj._topologicalFeatures['diagrams'] = []

    # save the new object as subject fileType joined remove the rips diagrams
    #  that is overwrite them and note which prePI was used (though not preferable)
    if save:
        substr = [str(i) for i in subject]
        pickleString = "subject_" + "-".join(substr) + "_" + fileType + "_joinedPI.pickle"
        with open(pickleString, "wb") as f:
            pickle.dump(joinedObj, f)

    return joinedObj


def convertAveragedTDA(fileType="perceptionTraining"):
    if fileType == "perceptionTraining":
        typeNum = 2
    elif fileType == "perceptionTest":
        typeNum = 1
    elif fileType == "imageryTest":
        typeNum = 0
    else:
        raise Exception("fileType specified but not one of perceptionTraining, perceptionTest, or imageryTest.")
    for subj in range(1, 6):
        event_dict = dict()
        # get the number of runs
        m1 = masker.maskSubject(subj, "VC")
        fileMatches = [int(f[-2:]) for f in os.listdir("fMRIFullData/sub-0" + str(subj) + "/")
                       if re.match("ses-" + fileType + "*", f)]
        runNum = max(fileMatches)
        for run in range(1, runNum + 1):
            # get the number of trials
            fileMatches = [int(f[-14:-12]) for f in
                           os.listdir("fMRIFullData/sub-0" + str(subj) + "/ses-" + fileType + "%02i"
                                      % run + "/func/")
                           if re.match(
                    "sub-0" + str(subj) +
                    "_ses-" + fileType + "0" + str(run) +
                    "_task-perception_run-0*", f)]
            trialNum = max(fileMatches)
            for trial in range(1, trialNum + 1):
                # set-up a pickle string
                #print("Running ", pickleString)

                # read and convert the given nii.gz file with mask passing in the events as labels
                # open data
                #data = loader.loaderSpecific(subj, 0, run, trial)
                # cutoff the first few timepoints that are only rest, same with the last few
                #data = data[:, :, :, 10:(data.shape[3] - 3)]
                # fit rips and object with labels for class
                events = pd.read_csv("fMRIFullData/sub-0" + str(subj) +
                                     "/ses-" + fileType + "0" + str(run) +
                                     "/func/sub-0" + str(subj) +
                                     "_ses-" + fileType + "0" + str(run) +
                                     "_task-perception_run-%02i" % trial + "_events.tsv", sep='\t')

                for i in range(1, len(events['stimulus_name'])-1):
                    label = events['stimulus_name'][i]
                    if label not in event_dict.keys():
                        event_dict[label] = [(run, trial, 11 + (i-1)*3, (i-1)*3 + 14)]
                    else:
                        event_dict[label].append((run, trial, 11 + (i-1)*3, (i-1)*3 + 14))

        dataFull = None

        all_labels = []
        for label in event_dict.keys():
            label_data = None
            all_labels.append(label)
            print(label)
            for run, trial, start, stop in event_dict[label]:
                data = loader.loaderSpecific(subj, typeNum, run, trial)
                print(start, ":", stop)
                #print(data.shape)
                if label_data is None:
                    label_data = data[:,:,:, start:stop].copy()
                else:
                    label_data += data[:,:,:, start:stop].copy()
            label_data /= len(event_dict[label])
            if dataFull is None:
                dataFull = label_data
            else:
                dataFull = np.append(dataFull, label_data, axis=3)

    # take from here






        tf = TopologicalfMRI(dataFull, all_labels,
                             m1, nTime=3, keepRange=(0, 3),
                             saveMask=False)

                #tf.setLabels(np.array(events['category_name'][1:(events['category_name'].shape[0] - 1)]))
                #print(tf._labels)  # confirm labels are correct

                # write the pickle file in top directory
        pickleString = ("subject_" + str(subj) + "_" + fileType +
                        "_VC_CORR_AVG_SHIFT.pickle")
        with open(pickleString, "wb") as f:
            pickle.dump(tf, f)