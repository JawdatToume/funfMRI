"""
Author : Srinjoy bhuiya
Motivation : This code uses the sparse linear regressor and runs them on the fmri data

TODO: Read the fMRI Data
TODO: Create batches of fMRI vectors(Input data) and 1 value of CNN feature vector(prediction
        targets) over all the images showns(aka fMRI reading per timestep)
TODO : train the model over all the features to be trained for

"""
import pickle
#import fMRITDA
import masker, loader
import pandas as pd
import re,os
def trainLinReg(fileType="perceptionTest"):
    for subj in range(1, 6): # Looping through all the subjects
        # get the number of runs
        m1 = masker.maskSubject(subj, "VC")
        fileMatches = [int(f[-2:]) for f in os.listdir("/home/srinjoy/PycharmProjects/funfMRI/fMRIFullData/sub-0" + str(subj) + "/")
                       if re.match("ses-" + fileType + "*", f)]
        runNum = max(fileMatches)
        for run in range(1, runNum + 1):
            # get the number of trials
            fileMatches = [int(f[-14:-12]) for f in
                           os.listdir("/home/srinjoy/PycharmProjects/funfMRI/fMRIFullData/sub-0" + str(subj) + "/ses-" + fileType + "0"
                                      + str(run) + "/func/")
                           if re.match(
                    "sub-0" + str(subj) +
                    "_ses-" + fileType + "0" + str(run) +
                    "_task-perception_run-0*", f)]
            trialNum = max(fileMatches)

            # print(fileMatches)
            # print(runNum)
            # print(trialNum)
            for trial in range(1, trialNum + 1):
                # set-up a pickle string
                pickleString = ("subject_" + str(subj) + "_" + fileType + "_run_" + str(run) + "_trial_" + str(trial) +
                                "_VC.pickle")
                print("Running ", pickleString)

                # read and convert the given nii.gz file with mask passing in the events as labels
                # open data
                data = loader.loaderSpecific(subj, 2, run, trial)

                if isinstance(data, str):
                    print("No file available (Not an error)")
                    continue

                print("Trial Num: ",trial)
                # cutoff the first few timepoints that are only rest, same with the last few
                data = data[:, :, :, 10:(data.shape[3] - 3)]
                # fit rips and object with labels for class
                events = pd.read_csv("/home/srinjoy/PycharmProjects/funfMRI/fMRIFullData/sub-0" + str(subj) +
                                     "/ses-" + fileType + "0" + str(run) +
                                     "/func/sub-0" + str(subj) +
                                     "_ses-" + fileType + "0" + str(run) +
                                     "_task-perception_run-%02i" % trial + "_events.tsv", sep='\t')
                print(data)
                print(events)

                # TODO: use the fMRI data received and event file to find the feature and fmri pair 
                # tf = fMRITDA.TopologicalfMRI(data, np.array(events['stimulus_name'][1:(events['stimulus_name'].shape[0] - 1)]),
                #                      m1, nTime=3, keepRange=(0, 2),
                #                      saveMask=False)
                #
                # # tf.setLabels(np.array(events['stimulus_name'][1:(events['stimulus_name'].shape[0] - 1)]))
                # print(tf._labels)  # confirm labels are correct
                #
                # # write the pickle file in top directory
                # with open(pickleString, "wb") as f:
                #     pickle.dump(tf, f)

if __name__ == "__main__":
    x=trainLinReg()


