import nibabel as nib
import nilearn as nil
import nilearn.plotting as plotting
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def maskSubject(subjectNo, maskName = 'VC'):
    maskStrings = {'V1': ['V1d', 'V1v'],
                   'V2': ['V2d', 'V2v'],
                   'V3': ['V3d', 'V3v'],
                   'V4': ['hV4'],
                   'FFA': ['FFA'],
                   'LOC': ['LOC'],
                   'PPA': ['PPA'],
                   'VC': ['V1d', 'V1v', 'V2d', 'V2v',
                          'V3d', 'V3v', 'V3d', 'V3v',
                          'hV4', 'HVC', 'FFA', 'LOC',
                          'PPA']}

    fullMask = None
    for maskName in maskStrings[maskName]:
        for side in ['LH', 'RH']:
            filepath = os.path.join("fMRIFullData",
                       os.path.join("sourcedata",
                       os.path.join("sub-0" + str(subjectNo),
                       os.path.join("anat",
                       os.path.join("sub-0" + str(subjectNo) +
                                    "_mask_" + side + "_" + maskName +
                                    ".nii.gz")))))

            file = nib.load(filepath)
            if fullMask is None:
                fullMask = file.get_fdata()
            else:
                fullMask += file.get_fdata()

    fullMask = fullMask > 0

    return fullMask

def maskSubjectData(maskData, subjectData):
    newData = subjectData.copy()
    for timepoint in range(subjectData.shape[3]):
            newData[:, :, :, timepoint] = newData[:, :, :, timepoint] * maskData

    return newData

# will flatten the data too
# def maskSubjectDataTrim(maskData, subjectData):
