import nibabel as nib
import nilearn as nil
import nilearn.plotting as plotting
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import masker
import torch
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
    
    regress(x_train, y_train, x_test)

def regress(x_train, y_train, x_test):
    model = LinearRegression()

    print(x_train.shape)
    print(y_train.shape)
    
    model.fit(x_train, y_train)

    y_prediction = model.predict(x_test)

    # y_test_list = []
    # y_prediction_list = []
    # for i in range(len(y_test)):
    #     y_test_list.append(y_test[i])
    #     y_prediction_list.append(y_prediction[i])
    # y_test_list = np.array(y_test_list)
    # y_prediction_list = np.array(y_prediction_list)
    # print(np.corrcoef(y_test_list, y_prediction_list))   
    # TODO: Calculate correlations between the images and find the one that is most matching the category :)
    #  

if __name__ == "__main__":
    main()