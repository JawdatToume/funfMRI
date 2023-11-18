import nibabel as nib
import nilearn as nil
import nilearn.plotting as plotting
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import masker
import torch

# https://gi    thub.com/KamitaniLab/slir
from slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression
import h5py

'''Function:
regress(x_train, y_train, x_test, y_test)
You can call regress separately or use the command line 
./regress.py x_train y_train x_test y_test
If you don't input anything it just assumes 4 random 1000 x 1 matrices

returns the linear regressors and the correlation coefficient'''
class Regressor:
    def fit(self, flattened_data, feature_vector):
        self.linregs = []
        for i in range(feature_vector.shape[1]):
            self.linregs.append(SparseLinearRegression(n_iter=100))
            print(flattened_data.shape,feature_vector[i,:].shape)
            #
            self.linregs[i].fit(flattened_data, feature_vector[:,i])
    
    def predict(self, test_data):
        predicted_features = []
        for regressor in self.linregs:
            predicted_features.append(regressor.predict(test_data)[0])
        predicted_features = np.array(predicted_features)

        return predicted_features 

    def evaluate(self, predicted_features, actual_value):
        correlation = np.corrcoef(actual_value.reshape(actual_value.shape[0],), predicted_features)

        return correlation

# def main():

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
#     if(len(sys.argv) > 1):
#         regress(np.random.random(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3]))
#     else:
#         regress(np.random.random((1000,1)), np.random.random((1000,5)), np.random.random((1000,1)), np.random.random((1000,5)))


# if __name__ == "__main__":
#     main()