from pathos.multiprocessing import ProcessingPool as Pool
import nibabel as nib
import nilearn as nil
import nilearn.plotting as plotting
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import masker
# import torch

# https://github.com/KamitaniLab/slir
from slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression
import h5py
# from multiprocessing import Pool

'''Function:
regress(x_train, y_train, x_test, y_test)
You can call regress separately or use the command line 
./regress.py x_train y_train x_test y_test
If you don't input anything it just assumes 4 random 1000 x 1 matrices

returns the linear regressors and the correlation coefficient'''
class regressor:
    def fit(self, flattened_data, feature_vector):
        #self.linregs = [SparseLinearRegression(n_iter=20) for _ in range(feature_vector.shape[1])]
        self._norm_mean = np.mean(flattened_data, axis=0)
        self._norm_scale = np.std(flattened_data, axis=0, ddof=1)

        for i in range(len(self._norm_mean)):
            if self._norm_scale[i] >= 1e-12:
                try:
                    flattened_data[:, i] = (flattened_data[:,i]-self._norm_mean[i]) / self._norm_scale[i]
                except:
                    np.delete(flattened_data, i, 1)
            else:
                np.delete(flattened_data, i, 1)
        def fitData(i):
            temp = SparseLinearRegression(n_iter=200)
            temp.fit(flattened_data, feature_vector[:,i])
            return temp
            #self.linregs[i].fit(flattened_data, feature_vector[:,i])

        with Pool(40) as p:
            self.linregs = p.map(fitData, [i for i in range(feature_vector.shape[1])])
            
        # return None
        #for i in range(feature_vector.shape[1]):
        #    print(i, end=" ")
            # self.linregs.append(SparseLinearRegression(n_iter=10))
            # print(feature_vector[i,:].shape)
        #    self.linregs[i].fit(flattened_data, feature_vector[:,i])
    
    def predict(self, test_data):
        predicted_features = []
        for regressor in self.linregs:  # naming!!!
            predicted_features.append(regressor.predict(test_data)[0])
        predicted_features = np.array(predicted_features)
        return predicted_features

    def evaluate(self, predicted_features, result):
        correlation = np.corrcoef(result, predicted_features)

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
