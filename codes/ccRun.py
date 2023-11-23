import fMRITDA
from labeler import labelsToFeatures
from regresser import regressor
import pickle
i = 1
trainData = fMRITDA.joinTDA(subject=[1], fileType="perceptionTraining")
#for i in range(1, 2):
#for l_num in [22, 11, 17, 1, 20, 4, 7, 9]:
for l_num in [22]:
    print("Running Subject: 1")
    
    
    print("Extracting Data", end=" - ")
    trainIn, trainLabels = trainData.getData()

    print("Getting Training Labels", end=" - ")
    trainLabels = labelsToFeatures(trainLabels[0:(trainIn.shape[0])], layer=l_num)
    print("Training Regresser")
    subReg = regressor()
    subReg.fit(trainIn, trainLabels)

    print("Saving Trained Regresser")
    pickleString = "subject_" + "1" + "_Layer_"+str(l_num)+"_Regresser_200iter.pickle"
    with open(pickleString, "wb") as f:
        pickle.dump(subReg, f)
