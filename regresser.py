import nibabel as nib
import nilearn as nil
import nilearn.plotting as plotting
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import masker
import torch

# https://github.com/KamitaniLab/slir
from slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression
import h5py

def main():
    fmridata = masker.maskSubject(1)
    #h5py.File("Subject1_ImageNetTraining.h5", 'r')
    #h5py.File("Subject1_ImageNetTest.h5", 'r')

    x_train = fmridata[:][:][:40]
    x_test = fmridata[:][:][40:]
    #y_train = h5py.File("Subject1.h5", 'r') 
    #y_train = np.array(y_train["dataset"][:])
    y_train = torch.load(os.path.join("convnext_train", "01518878_5958.pt"))
    # A. open folder with pytorch so we can extract the features from the .pt files.
    # TODO: Figure out how to identify the (technical term) thingies.
    
    regress(np.random.random((1000,1)), np.random.random((1000,5)), np.random.random((1000,1)), np.random.random((1000,5)))

def regress(flattened_data, feature_vector, test_data, result):
    linregs = []
    for i in range(len(feature_vector)):
        linregs.append(SparseLinearRegression(n_iter=100))
        linregs[i].fit(flattened_data.T, feature_vector[i][:].reshape(1, feature_vector.shape[1]))
    
    predicted_features = []
    for regressor in linregs:
        predicted_features.append(regressor.predict(test_data.T)[0])
    predicted_features = np.array(predicted_features)
    print(result)
    print(predicted_features)
    print(np.corrcoef(result.reshape(result.shape[0],), predicted_features))

if __name__ == "__main__":
    main()