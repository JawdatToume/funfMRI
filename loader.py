import nibabel as nib
import nilearn as nil
import nilearn.plotting as plotting
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import masker
import regresser
import labeler

from nilearn import image

# i wanna make this prettier and more versatile eventually but I at least got fmri stuff opened os that;s good!!
# is prettier and more versatile tbh i liek this file

'''Instructions for running:
    loaderAll: normalizes and gets data for all subjects (don't do this lol)
    loaderSubject(subjectNo): normalizes and gets data for the subject (1 <= subjectNo <= 5)
    loaderSpecific(subjectNo, dataType, dataNo, testNo): gets the specific matrix from the specified subject, dataType (0 = imageryTest, 1 = perceptionTest, 2 = perceptionTraining), data number, and test number to give one matrix
    
    Note: if running in command line, it just takes the one with the most number of parameters that could be accounted for:
        0 inputs -> loaderAll() (can also be run with loader.py all), 
        1-3 inputs -> loaderSubject(sys.argv[1]),
        4+ -> loaderSpecific(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])'''

#https://brainiak.org/tutorials/02-data-handling/
def main():
    if len(sys.argv) == 1 or sys.argv[1] == "all":
        loaderAll()
    elif len(sys.argv) < 5:
        loaderSubject(sys.argv[1])
    else:
        loaderSpecific(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    

def loaderAll():
    print("running loaderAll...")
    points = []
    mask = masker.maskSubject(1, "V1")

    for subjectNo in range(1,6):
        points.append([])
        filepath = os.path.join("fMRIFullData",os.path.join("sub-0" + str(subjectNo)))
        for dataTypes in os.listdir(filepath)[1:]:
            filepath2 = os.path.join(filepath, os.path.join(dataTypes, "func"))
            for dataMatrices in os.listdir(filepath2):
                filepath3 = os.path.join(filepath2, dataMatrices)
                if filepath3[-1] == "z":
                    file = nib.load(filepath3)

                    data = file.get_fdata()
                    data = masker.maskSubjectData(mask, data)

                    points[subjectNo-1].append(data)
                #pickle.dump(points, filepath3.replace("\\",""))
    
    for subjectNo in range(0,5):
        currSubject = points[subjectNo]
        complete = []
        for point in currSubject:
            complete.extend(point.tolist())
        complete = np.array(complete)
        norm = np.linalg.norm(complete)

        for point in range(len(currSubject)):
            currSubject[point] /= norm

    print("done")
    return points

def loaderSubject(subjectNo):
    model = regresser.regressor()
    labels = labeler.labelsToCSV()
    currkey = 0
    keys = labels[0]
    print(keys)
    features = labels[1]
    print(features)
    print("running loaderSubject...")
    #reader = open("fMRIFullData-sub-01-ses-imageryTest01-func.pkl", 'rb')
    points = []#pickle.load(reader)

    filepath = os.path.join("fMRIFullData",os.path.join("sub-0" + str(subjectNo)))
    for dataTypes in os.listdir(filepath)[1:]:
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

                model.fit(data, features[currkey])
                currkey+=1

                points.append(data)
    
                #print(data.shape)
        #np.save(writefile, points) #We are constantly getting MemoryErrors, I'm currently trying to look into making things faster through this

    model.save("modelSub1.pkl")
    complete = []
    

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

def loaderSpecific(subjectNo, dataType, dataNo, testNo):
    print("running loaderSpecific...")
    if dataType == 0:
        testType = "imageryTest"
        testType2 = "imagery"
    elif dataType == 1:
        testType = "perceptionTest"
        testType2 = "perception"
    else:
        testType = "perceptionTraining"
        testType2 = "perception"
    # print(f"{(lambda a: '0'+str(a) if a<10 else str(a))(testNo)}")
    try:
        filepath = os.path.join(f"./fMRIFullData",os.path.join("sub-0" + str(subjectNo), os.path.join("ses-"+testType+"0"+str(dataNo), os.path.join("func", "sub-0"+str(subjectNo)+"_ses-"+testType+"%02i"%(dataNo)+"_task-"+testType2+"_run-"+f"{(lambda a: '0'+str(a) if a<10 else str(a))(testNo)}"+"_bold.nii.gz"))))
        # print(filepath)
        file = nib.load(filepath)

        data = file.get_fdata()
        print("done")
    except:
        print("Either Loading of all the sessions completed or some error in the try block ")
        return "FIN"
    # print(data)

    # #fig, axes = plt.subplots(1,3, figsize=(15,35))
    # #axes = np.ravel(axes)
    # print(data.shape)
    # plt.imshow(data[:,:,36])#.T, cmap='gray', origin='lower') # Saggital
    # plt.show()#axes[0].set_title("Saggital")
    # plt.imshow(data[:,:,2])#.T, cmap='gray', origin='lower') # Saggital
    # plt.show()#axes[0].set_title("Saggital")

    return data

    #plotting.plot_matrix(data)

if __name__ == "__main__":
    main()