import torch
import numpy as np
import pandas as pd
import pickle

# code to turn labels into feature vectors for particular layers

def labelsToCSV(layer=22):
    layer_lab = "Output Layer " + str(layer)
    labelDict = torch.load("featureData/test_train_img_features/kaggle/working/alexnet_dict_train.pt")
    keys = []
    featureArray = np.empty((len(labelDict.keys()), 1000))
    i = 0
    for key in labelDict.keys():
        keys.append(key)
        trimKey = key
        featureArray[i,:] = labelDict[trimKey][layer_lab].flatten()
        i += 1

    with open("featureData/labels_train_Layer"+str(layer)+".pickle", "wb") as f:
        pickle.dump((keys, featureArray), f)

    return (keys, featureArray)

def labelsToFeatures(labels, layer=22):
    with open("featureData/labels_train_Layer"+str(layer)+".pickle", "rb") as f:
        labelObj = pickle.load(f)

    # labelDict = torch.load("featureData/test_train_img_features/kaggle/working/alexnet_dict_train.pt")
    featureArray = np.empty((len(labels), 1000))
    i = 0
    for key in labels:
        trimKey = key[1:]
        indKey = labelObj[0].index(trimKey)
        if indKey != -1:
            featureArray[i,:] = labelObj[1][indKey, :]
        i += 1

    return featureArray

if __name__ == "__main__":
    labelsToCSV()
