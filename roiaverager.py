import masker
import loader
import numpy as np
import nibabel as nib
import os
import pandas as pd
import labeler
import pickle
import math
import regresser
import fMRITDA
import scipy
import csv

def main():
    #model = loader.loaderSubject(1)
    #for i in range(1):
    for i in [22, 11, 17, 1, 4, 7, 9, 20]: #
        # model = regresser.regressor()
        # model.load("modelSub%sUpdate.pkl" %(str(i+1)))
    #    for j in range(1,3):
            # writer = open("results%s%sImage.csv" %(str(i), str(j)), 'a')
            # results = csv.writer(writer)
            # testSubject(model, j+1, results, "imagery")
            # writer.close()
            # writer = open("results%s%sPerceptionTest.csv" %(str(i), str(j)), 'a')
            # results = csv.writer(writer)
            # testSubject(model, j+1, results, "perceptionTest")
            # writer.close()
            # writer = open("results%s%sPerceptionTraining.csv" %(str(i), str(j)), 'a')
            # results = csv.writer(writer)
            # testSubject(model, j+1, results, "perceptionTraining")
        
        try:
            model = regresser.regressor()
            model.load("modelSubTrue1LayerperceptionTraining%s.pkl" %(str(i)))
        except:
            model = trainSubject(1, 'perceptionTraining', i)
            #model = trainAVG(1, 'perceptionTraining', i)
        writer = open("resultsAVG04traintestLayers.csv", 'a')
        results = csv.writer(writer)
        testSubject(model, 1, results, 'perceptionTest', i)
        #testAVG(model, 5, results, 'perceptionTest', i)
        writer.close()

    # model = regresser.regressor()
    # model.load("modelSub1.pkl")
    #testSubject(model, 2)
    # writer = open("subject1V1.pkl", 'wb')
    # pickle.dump(points, writer)
    # writer.close()
    #print(data.mean())
def trainAVG(subjectNo, trainType, layer):
    model = regresser.regressor()
    labels = labeler.labelsToCSV(layer)
    feature = labels[1]
    features = []
    currkey = 0
    filepath = os.path.join("fMRIFullData",os.path.join("sub-0" + str(subjectNo)))
    for dataTypes in os.listdir(filepath)[1:]:
        if dataTypes.find(trainType) == -1:
            currkey+=1
            continue
        print(dataTypes)
        filepath2 = os.path.join(filepath, os.path.join(dataTypes, "func"))
        writefile = filepath2.replace("\\", "-")
        writefile += '.pkl'
        for dataMatrices in os.listdir(filepath2):
            print(dataMatrices)
            filepath3 = os.path.join(filepath2, dataMatrices)
            if filepath3[-1] == "z":
                features.append(feature[currkey])
    features = np.array(features)
    feature = []
    for i in range(features.shape[1]):
        feature.append(np.mean(features[:,i]))

    data = fMRITDA.convertAveragedTDA(trainType, subjectNo)
    data = average(data, subjectNo)
    model.fit(data, feature)
    model.save("modelSubAVG%sUpdateLayer%s%s.pkl" %(str(subjectNo), str(trainType), str(layer)))
    return model

def testAVG(model, subjectNo, csvwrite, testType, layer):
    corrcofs = [0, layer]
    data = fMRITDA.convertAveragedTDA(testType, subjectNo)
    labels = labeler.labelsToCSV(layer)
    feature = labels[1]
    features = []
    currkey = 0
    filepath = os.path.join("fMRIFullData",os.path.join("sub-0" + str(subjectNo)))
    for dataTypes in os.listdir(filepath)[1:]:
        if dataTypes.find(trainType) == -1:
            currkey+=1
            continue
        print(dataTypes)
        filepath2 = os.path.join(filepath, os.path.join(dataTypes, "func"))
        writefile = filepath2.replace("\\", "-")
        writefile += '.pkl'
        for dataMatrices in os.listdir(filepath2):
            print(dataMatrices)
            filepath3 = os.path.join(filepath2, dataMatrices)
            if filepath3[-1] == "z":
                features.append(feature[currkey])
    features = np.array(features)
    feature = []
    for i in range(features.shape[1]):
        feature.append(np.mean(features[:,i]))
    data = average(data, subjectNo)
    result = model.predict(data)
    for i in range(result.shape[1]):
        corrcofs.append(scipy.stats.pearsonr(result[:,i].reshape(result.shape[0],), feature[:,i].reshape(feature.shape[0],)).statistic)
    print(corrcofs)
    csvwrite.writerow(corrcofs)


def average(data, subjectNo):
    newdata = []
    rois = ['V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA']
    for roi in rois:
        thisTime = []
        for timepoint in range(data.shape[3]):
            mask = masker.maskSubject(subjectNo, roi)
            maskedData = masker.maskSubjectData(mask, data)
            print(maskedData.shape)
            thisTime.append(np.mean(maskedData[:,:,:,timepoint]))
        newdata.append(thisTime)


    return np.array(newdata)

def trainSubject(subjectNo, trainType, layer):
    model = regresser.regressor()
    labels = labeler.labelsToCSV(layer)
    currkey = 0
    keys = labels[0]
    datas = []
    featureList = []
    print(keys)
    features = labels[1]
    print(features)
    print("running loaderSubject...")
    #reader = open("fMRIFullData-sub-01-ses-imageryTest01-func.pkl", 'rb')
    points = []#pickle.load(reader)

    filepath = os.path.join("fMRIFullData",os.path.join("sub-0" + str(subjectNo)))
    for dataTypes in os.listdir(filepath)[1:]:
        if dataTypes.find(trainType) == -1:
            currkey+=1
            continue
        print(dataTypes)
        filepath2 = os.path.join(filepath, os.path.join(dataTypes, "func"))
        writefile = filepath2.replace("\\", "-")
        writefile += '.pkl'
        for dataMatrices in os.listdir(filepath2):
            print(dataMatrices)
            filepath3 = os.path.join(filepath2, dataMatrices)
            if filepath3[-1] == "z":
                file = nib.load(filepath3)

                data = file.get_fdata()
                data = average(data, subjectNo)
                datas.append(data.tolist())
                featureList.append(features[currkey].tolist())
                currkey+=1

                points.append(data)
    
                #print(data.shape)
        #np.save(writefile, points) #We are constantly getting MemoryErrors, I'm currently trying to look into making things faster through this

    data = np.array(datas)
    feature = np.array(featureList)
    print(data.shape)
    print(feature.shape)
    model.fit(data.reshape(data.shape[0],data.shape[1]), feature.reshape(feature.shape[0],feature.shape[1]))
    model.save("modelSub%sUpdateLayer%s%s.pkl" %(str(subjectNo), str(trainType), str(layer)))
    

    # for i in range(len(points)):
    #     complete.extend(points[i].tolist())
    #     points[i] = None
    #     print(i)
    # complete = np.array(complete)
    #norm_mean = np.mean(complete)
    #norm_std = np.std(complete)

    # print("")
    # for point in range(len(points)):
    #     points[point] = (points[point] - norm_mean) / norm_std
    
    # print("done")
    return model

def testSubject(model, subjectNo, csvwrite, testtype="", layer=22):
    labels = labeler.labelsToCSV(layer)
    corrcofs = [0, layer]
    results = []
    featureList = []
    currkey = 0
    keys = labels[0]
    features = labels[1]
    print("running testSubject...")
    results = []

    filepath = os.path.join("fMRIFullData",os.path.join("sub-0" + str(subjectNo)))
    for dataTypes in os.listdir(filepath)[1:]:
        if dataTypes.find(testtype) == -1:
            currkey+=1
            continue
        print(dataTypes)
        filepath2 = os.path.join(filepath, os.path.join(dataTypes, "func"))
        writefile = filepath2.replace("\\", "-")
        writefile += '.pkl'
        for dataMatrices in os.listdir(filepath2):
            print(dataMatrices)
            filepath3 = os.path.join(filepath2, dataMatrices)
            if filepath3[-1] == "z":
                file = nib.load(filepath3)

                data = file.get_fdata()
                data = average(data, 2)

                result = model.predict(data)
                print(result.shape)
                feature = features[currkey].reshape(features[currkey].shape[0],1)
                print(feature.shape)
                results.append(result.tolist())
                featureList.append(feature.tolist())
                currkey+=1
    
                #print(data.shape)
        #np.save(writefile, points) #We are constantly getting MemoryErrors, I'm currently trying to look into making things faster through this

    result = np.array(results)
    feature = np.array(featureList)
    print(result.shape)
    print(feature.shape)
    for i in range(result.shape[1]):
        corrcofs.append(scipy.stats.pearsonr(result[:,i].reshape(result.shape[0],), feature[:,i].reshape(feature.shape[0],)).statistic)
    print(corrcofs)
    csvwrite.writerow(corrcofs)
    

    # for i in range(len(points)):
    #     complete.extend(points[i].tolist())
    #     points[i] = None
    #     print(i)
    # complete = np.array(complete)
    #norm_mean = np.mean(complete)
    #norm_std = np.std(complete)

    # print("")
    # for point in range(len(points)):
    #     points[point] = (points[point] - norm_mean) / norm_std
    
    # print("done")
    return model

if __name__ == "__main__":
    main()  