import nibabel as nib
import nilearn as nil
import nilearn.plotting as plotting
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import masker
import torch
import pickle
import math

# https://github.com/KamitaniLab/slir
from slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression
import h5py

'''Function:
regress(x_train, y_train, x_test, y_test)
You can call regress separately or use the command line 
./regress.py x_train y_train x_test y_test
If you don't input anything it just assumes 4 random 1000 x 1 matrices

returns the linear regressors and the correlation coefficient'''

class regressor:
    def __init__(self):
        self.linregs = []

    def fit(self, flattened_data, feature_vector):
        self.linregs = []
        for i in range(feature_vector.shape[1]):
            while(len(self.linregs) <= i):
                self.linregs.append(SparseLinearRegression(n_iter=100, prune_mode=0, prune_threshold=0, minval=0.01, converge_threshold=0.01))
            #print(feature_vector[:,i].shape)

            print(flattened_data.shape)
            yvalue = feature_vector[:,i].tolist()
            #print(yvalue.shape)
            for j in range(len(yvalue)):
                if math.isnan(yvalue[j]) or math.isinf(yvalue[j]):
                    yvalue.pop(j)
                    flattened_data.pop(j)
            yvalue = np.array(yvalue)
            try:
                self.linregs[i].fit(flattened_data, yvalue.T)
            except:
                pass

    def save(self, path):
        saver = open(path, 'wb')
        pickle.dump(self.linregs, saver)
        saver.close()
    
    def load(self, path):
        loader = open(path, 'rb')
        self.linregs = pickle.load(loader)
    
    def predict(self, test_data):
        predicted_features = []
        prediction = []
        for regressor in self.linregs:
            test_data = test_data.reshape(test_data.shape[0],1)
            try:
                prediction = regressor.predict(test_data.T)
                predicted_features.append(prediction)
            except:
                pass
        predicted_features = np.array(predicted_features)

        return predicted_features 


    def evaluate(self, predicted_features, actual_value):
        correlation = np.corrcoef(actual_value, predicted_features)

        return correlation

def main():

#     # fmridata = masker.maskSubject(1)
#     # #h5py.File("Subject1_ImageNetTraining.h5", 'r')
#     # #h5py.File("Subject1_ImageNetTest.h5", 'r')

#     # x_train = fmridata[:][:][:40]
#     # x_test = fmridata[:][:][40:]
#     # #y_train = h5py.File("Subject1.h5", 'r') 
#     # #y_train = np.array(y_train["dataset"][:])
#     # y_train = torch.load(os.path.join("convnext_train", "01518878_5958.pt"))
#     # A. open folder with pytorch so we can extract the features from the .pt files.
#     # TODO: Figure out how to identify the (technical term) thingies.
    object = regressor()
    object.fit(np.random.random((1000,1)), np.random.random((1000,5)))
    features = object.predict(np.random.random((1000,1)))
    corrmat = object.evaluate(features, np.random.random((1000,5)))


if __name__ == "__main__":
    main()