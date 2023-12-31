"""
Author: Srinjoy Bhuiya

"""
import numpy as np
from sklearn import preprocessing as pre
from slir import SparseLinearRegression
import os
import pickle
from pathos.multiprocessing import ProcessingPool as Pool

class Slir_regressor:
    def __init__(self, n_iter=200, minval=1.0e-15,
                 prune_mode=1, prune_threshold=1.0e-10,
                 converge_min_iter=100, converge_threshold=1.0e-10,
                 verbose=True, verbose_skip=10):
        self.linregs = []
        self.predicted_features = []
        self.n_iter = n_iter
        self.prune_mode = prune_mode
        self.minval = minval
        self.prune_threshold = prune_threshold
        self.converge_min_iter = converge_min_iter
        self.converge_threshold = converge_threshold
        self.verbose = verbose
        self.verbose_skip = verbose_skip

    def fit(self, X_train, Y_train):

        def fitData(i):
            print("For feature :", (i + 1), "out of :", Y_train.shape[1])
            temp = SparseLinearRegression(n_iter=self.n_iter)
            temp.fit(X_train, Y_train[:,i])
            return temp


        with Pool(40) as p:
            self.linregs = p.map(fitData, [i for i in range(Y_train.shape[1])])
    def save(self, path):
        pickle.dump(self.linregs, path)

    def predict(self, X_test_data):
        for index, reg in enumerate(self.linregs):
            predicted_feature = reg.predict(X_test_data)
            self.predicted_features.append(predicted_feature)
        self.predicted_features = np.transpose(np.array(self.predicted_features), (1, 0))

        return self.predicted_features

    def evaluate(self, Y_test):

        correlation = np.corrcoef(Y_test, self.predicted_features)

        return correlation


def main(dir="/srinjoy_data/SUB_WISE_TRAIN_TEST_SPLIT_DATA/",
         train_include=[1],
         test_include=[1]):
    subs_dirs = sorted(os.listdir(dir))
    print(subs_dirs)
    subjects = [x for x in range(1, 6)]

    for sub in subjects:
        per_sub_dir = os.path.join(dir, subs_dirs[sub - 1])
        per_layer_lin_reg = []
        per_layer_predicted_y_test = []
        per_layer_corr_mat = []
        if sub in train_include:
            print("For subject :", sub, "out of :", len(subjects))
            x_train_file = open(per_sub_dir + f"/subject_{sub}_train_X_for_ann.pkl", "rb")
            y_train_file = open(per_sub_dir + f"/subject_{sub}_train_Y_for_ann.pkl", "rb")

            X_train_per_sub = pickle.load(x_train_file)

            Y_train_per_sub = np.transpose(pickle.load(y_train_file), (1, 0, 2))

            for layer in range(0, 8):
                print("For layer :", layer)
                X_train_per_sub_per_layer = pre.MinMaxScaler().fit_transform(X_train_per_sub)
                Y_train_per_sub_per_layer = Y_train_per_sub[layer]

                print(X_train_per_sub_per_layer.shape)
                print(Y_train_per_sub_per_layer.shape)
                per_layer_lin_reg.append(Slir_regressor(n_iter=2, verbose=True))
                per_layer_lin_reg[layer].fit(X_train_per_sub_per_layer, Y_train_per_sub_per_layer)
            pickle.dump(per_layer_lin_reg, open(per_sub_dir + f"/subject_{sub}_convnext_features_slir_regressor.pkl", "wb"))
            x_train_file.close()
            y_train_file.close()

        if sub in test_include:
            x_test_file = open(per_sub_dir + f"/subject_{sub}_test_X_for_ann.pkl", "rb")
            y_test_file = open(per_sub_dir + f"/subject_{sub}_test_Y_for_ann.pkl", "rb")
            X_test_per_sub = pickle.load(x_test_file)
            Y_test_per_sub = np.transpose(pickle.load(y_test_file), (1, 0, 2))

            for layer in range(0, 8):
                X_test_per_sub_per_layer = pre.MinMaxScaler().fit_transform(X_test_per_sub)
                Y_test_per_sub_per_layer = Y_test_per_sub[layer]

                print(X_test_per_sub_per_layer.shape)
                print(Y_test_per_sub_per_layer.shape)

                per_layer_predicted_y_test.append(per_layer_lin_reg[layer].predict(X_test_per_sub_per_layer))
                print("Predicted Features size: ", per_layer_predicted_y_test[layer].shape)
                print("Actual features size:", Y_test_per_sub_per_layer.shape)
                per_layer_corr_mat.append(per_layer_lin_reg[layer].evaluate(Y_test_per_sub_per_layer))
            pickle.dump(per_layer_predicted_y_test, open(per_sub_dir + f"/subject_{sub}_convnext_features_slir_predicted_y_test.pkl", "wb"))
            pickle.dump(per_layer_corr_mat, open(per_sub_dir + f"/subject_{sub}_convnext_features_slir_corr_mat.pkl", "wb"))
            pickle.dump(Y_test_per_sub, open(per_sub_dir + f"/subject_{sub}_convnext_features_slir_actual_y_test.pkl", "wb"))

            x_test_file.close()
            y_test_file.close()



if __name__ == "__main__":
    main()
