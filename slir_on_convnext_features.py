"""
Author: Srinjoy Bhuiya
TODO : Write this file to learn the convnext features from the fmri data using sparse lin regressor
"""
dir="/home/srinjoy/PycharmProjects/funfMRI/srinjoy_data/SUB_WISE_TRAIN_TEST_SPLIT_DATA/"
subjects=[x for x in range(1,6)]
train_include=[1,2]
test_include=[1,2]
for sub in subjects:
    if sub in train_include:
        continue
    if sub in test_include:
        continue
# import nibabel as nib
# import nilearn as nil
# import nilearn.plotting as plotting
# import os
# import sys
# import matplotlib.pyplot as plt
# import numpy as np
# import masker
# import torch
# import pickle
# from sklearn import preprocessing as pre
# from slir import SparseLinearRegression
# from sklearn.linear_model import LinearRegression
# import h5py
#
#
# class Slir_regressor:
#     def __init__(self):
#         self.linregs = []
#         self.predicted_features=[]
#
#     def fit(self, X_train, Y_train):
#         for i in range(X_train.shape[0]):
#             if (len(self.linregs) <= i):
#                 self.linregs.append(SparseLinearRegression(n_iter=200))
#             self.linregs[i].fit(X_train, Y_train[i])
#
#     def save(self, path):
#         pickle.dump(self.linregs, path)
#
#     def predict(self, test_data):
#         for index , regressor in enumerate(self.linregs):
#             predicted_feature= regressor.predict(test_data[index])
#             self.predicted_features.append(predicted_feature)
#         self.predicted_features=np.array(self.predicted_features)
#
#
#         return self.predicted_features
#
#     def evaluate(self, Y_test):
#         correlation = np.corrcoef(Y_test, self.predicted_features)
#
#         return correlation
#
#
# def main():
#
#     object = Slir_regressor()
#     object.fit(np.random.random((123, 10)), np.random.random((1000, 10)))
#     features = object.predict(np.random.random((123,1)))
#     corrmat = object.evaluate(features, np.random.random((1000,1)))
#
#
# if __name__ == "__main__":
#     main()

