"""
NOTE: DEPRECATED
Author : Srinjoy bhuiya
Motivation : This code uses the sparse linear regressor and runs them on the fmri data

TODO: Read the fMRI Data
TODO: Create batches of fMRI vectors(Input data) and 1 value of CNN feature vector(prediction
        targets) over all the images showns(aka fMRI reading per timestep)
TODO : train the model over all the features to be trained for

"""
import pickle
import datetime
import os
import re

import numpy as np
import pandas as pd
import torch

import loader
import masker


def trimFlatten(data, mask, nTime):
    # return the dataset flattened without the mask
    d1, d2, d3, tPoints = data.shape
    # flatten and trim data based on mask, expect 3 data points and trimmed rest periods
    flatData = np.empty((int(np.sum(mask)), nTime, int(tPoints / nTime)))
    keepDim = np.nonzero(mask)

    for i in range(0, tPoints, nTime):
        for j in range(0, nTime):
            flatData[:, j, int(i / nTime)] = np.reshape(data[keepDim][:, i + j], -1)

    return flatData


def trainLinReg(fileTypes=["perceptionTest","perceptionTraining"],
                train_img_features_cnn_pth="/home/srinjoy/PycharmProjects/funfMRI/img_features_cnn/convnext_features_train.pt",
                test_img_features_cnn_pth="/home/srinjoy/PycharmProjects/funfMRI/img_features_cnn/convnext_features_test.pt"):

    print(f"######RUN train_regressor_on_fmri_to_predict_cnn_features at time: {datetime.datetime.now()}","#########")

    count = 0
    for subj in range(1, 6):  # Looping through all the subjects
        # get the number of runs
        for fileType in fileTypes:
            mask_VC = masker.maskSubject(subj, "VC")
            fileMatches = [int(f[-2:]) for f in os.listdir("./fMRIFullData/sub-0" + str(subj) + "/")
                           if re.match("ses-" + fileType + "*", f)]
            runNum = max(fileMatches)
            train_img_features_cnn = torch.load(train_img_features_cnn_pth)
            test_img_features_cnn = torch.load(test_img_features_cnn_pth)

            print("For Subject Number:\t", subj)


            train_x_y_pairs_list = list()
            test_x_y_pairs_list= list()
            for run in range(1, runNum + 1):
                # get the number of trials
                fileMatches = [int(f[-14:-12]) for f in
                               os.listdir("./fMRIFullData/sub-0" + str(subj) + "/ses-" + fileType + "0"
                                          + str(run) + "/func/")
                               if re.match(
                        "sub-0" + str(subj) +
                        "_ses-" + fileType + "0" + str(run) +
                        "_task-perception_run-0*", f)]
                trialNum = max(fileMatches)

                # print(fileMatches)
                print("\t Runnum ::\t", runNum)
                # print(trialNum)
                for trial in range(1, trialNum + 1):
                    #set-up a pickle string
                    print("\t \t Trial Number :\t", trialNum)
                    pickleString = ("subject_" + str(subj) + "_" + fileType + "_run_" + str(run) + "_trial_" + str(trial) +
                                    "_VC.pickle",)
                    print("\t \t Running ", pickleString)

                    # read and convert the given nii.gz file with mask passing in the events as labels
                    # open data
                    data = loader.loaderSpecific(subj, 2, run, trial)

                    if isinstance(data, str):
                        print("\t \t No file available (Not an error)")
                        continue

                    # cutoff the first few timepoints that are only rest, same with the last few
                    data = data[:, :, :, 10:(data.shape[3] - 3)]
                    # fit rips and object with labels for class
                    events = pd.read_csv("./fMRIFullData/sub-0" + str(subj) +
                                         "/ses-" + fileType + "0" + str(run) +
                                         "/func/sub-0" + str(subj) +
                                         "_ses-" + fileType + "0" + str(run) +
                                         "_task-perception_run-%02i" % trial + "_events.tsv", sep='\t')
                    #
                    """
                    TODO: use the fMRI data received and event file to find the feature and fmri pair 
                    TODO 
                    You need to: 
                    1.  Open and mask the VC, this can be done by using Jawdat's loader to load specific areas and my masker 
                    2.  Remove any data that gets masked out, this is done on line 45 and 49 of fMRITDA; 
                    3.  Flatten the data, again this is done on line 49; 
                    4.  Train sparse linear models to predict feature values using this flattened data, 
                        this is just done using the code Jawdat gave (provided it works) the main issue is in joining 
                        the event files for image labels but that was given to you. 
                    5.  Testing will be done on perceptionTest and imageryTest data 
                        (I should ask you where the imagery features are for the CNN). 
                        This is to say that training is done on perceptionTraining data. 
                    6. Store the results ideally in a csv or tsv the columns should be organized as 
                                stimulus label (image id or category), 
                                run, 
                                rial (two numbers that separate the fMRI files run=folder, trial=file num), 
                                method (in your case newCNN but for mine TDA, and Jawdat would be avgROI), 
                                layer label (which layer of the CNN is being predicted), 
                                CNN (name just in case don't wanna mix up the two CNNs), 
                                1-1000 predictions 
                                (all 1000 predicted features the reason I propose this is because we can always get 
                                correlations from predicted features but we can't do the opposite so this is a just in case)
                    """
                    # print("Shape of the fMRI voxel data: ",data.shape)
                    # print("########EVENTS FILE FOR THE DATA###########")

                    events["stimulus_name"] = events["image_file"].str[:-5]

                    # data = data
                    labels = np.array(events['stimulus_name'][1:(events['stimulus_name'].shape[0] - 1)])
                    mask = mask_VC
                    nTime = 3

                    events.drop(events.tail(1).index, inplace=True)
                    events.drop(events.head(1).index, inplace=True)
                    # print(data.shape)
                    # print(events)

                    for index, row in events.iterrows():

                        image_stimulus = row["stimulus_name"][1:]

                        fmri_time_start = int(row['onset'] / 3) - 11
                        fmri_time_end = int((row['onset'] + 9) / 3) - 11
                        #print(fmri_time_start,fmri_time_end)
                        data_per_img_stimulus = data[:, :, :, fmri_time_start:fmri_time_end]
                        flatData_per_img_stimulus = np.transpose(
                            np.expand_dims((trimFlatten(data_per_img_stimulus, mask, nTime)).flatten("F"), 1))
                        print("\t \t !!!!!!###########Current file is for :",fileType[10:])
                        if(fileType[10:]=="Training"):
                            per_img_features = train_img_features_cnn[image_stimulus][0]  # TRAIN IMG SHOWN

                            train_x_y_pairs_list.append((flatData_per_img_stimulus,per_img_features,image_stimulus))
                            print("TRAIN added",(flatData_per_img_stimulus,per_img_features,image_stimulus),"to list")
                        elif(fileType[10:]=="Test"):

                            per_img_features = test_img_features_cnn[image_stimulus][0]  # TEST IMG SHOWN
                            test_x_y_pairs_list.append((flatData_per_img_stimulus, per_img_features, image_stimulus))
                            print("TEST added", (flatData_per_img_stimulus, per_img_features, image_stimulus), "to list")


                        # print(image_stimulus,flatData_per_img_stimulus,*per_img_features)
                        count += 1
            print("####################################")
            print("####################################")



            if(fileType[10:]=="Training"):
                print(len(train_x_y_pairs_list))
                with open(f"/home/srinjoy/PycharmProjects/funfMRI/sub_wise_lin_reg_data/{subj}_train.pkl", "wb") as train_save_f:
                    pickle.dump(train_x_y_pairs_list,train_save_f)
            elif (fileType[10:] == "Test"):
                print(len(test_x_y_pairs_list))
                with open(f"/home/srinjoy/PycharmProjects/funfMRI/sub_wise_lin_reg_data/{subj}_test.pkl", "wb") as test_save_f :
                    pickle.dump(test_x_y_pairs_list, test_save_f)


                        # for index, per_img_cnn_feature_name in enumerate(per_img_features.keys(include_nested=True)):
                        #     print("\t \t \t \t for feature name :\t ", per_img_cnn_feature_name, )
                        #     count += 1
                        #     per_img_cnn_features_layer = per_img_features[per_img_cnn_feature_name].detach().cpu().numpy()
                        #     # model=slir.SparseLinearRegression(n_iter=100)
                        #     # print(flatData_per_img_stimulus[:1000][:,0].shape,per_img_cnn_features_layer.shape)
                        #
                        #     # fit method takes X: (n_samples, n_features) to predict Y: (n_samples,1)
                        #     # THE number of samples must be equal and the shape must mathc like (422, 430) (422,)
                        #
                        #     print("\t \t \t \t X_train / X_test:\t", flatData_per_img_stimulus.shape, )
                        #     print("\t \t \t \t Y_train / Y_test:\t", per_img_cnn_features_layer.shape, )


                            # model.fit(flatData_per_img_stimulus, per_img_cnn_features_layer)
                            # predicted_data=model.predict(flatData_per_img_stimulus)
                            # print(predicted_data.shape)

                            # slir.evaluate(predicted_data,)

                    # flatData = trimFlatten(data, mask, nTime)

                    # print("####### LABELS :",labels)

                    # with pd.option_context('display.max_rows', None, 'display.max_columns',
                    #                        None):  # more options can be specified also
                    #     print(events)

                    # print(img_features_cnn)

                    # print(per_img_features)
                    # print(index, row,end='\t')
                    # print("\n")

                    # the flatdata just flattens the 4d fmri voxel data to 1d

                    # tf.setLabels(np.array(events['stimulus_name'][1:(events['stimulus_name'].shape[0] - 1)]))
                    # print(tf._labels)  # confirm labels are correct
    print("###################################", count,"######################################################")
    # TODO save the weights into a pickle file



if __name__ == "__main__":
    x = trainLinReg()



