import nibabel as nib
import nilearn as nil
import nilearn.plotting as plotting
import os
import matplotlib.pyplot as plt
import numpy as np

from nilearn import image

# i wanna make this prettier and more versatile eventually but I at least got fmri stuff opened os that;s good!!

#https://brainiak.org/tutorials/02-data-handling/

filepath = os.path.join("fMRIFullData",os.path.join("sub-01", os.path.join("ses-imageryTest01", os.path.join("anat", "sub-01_ses-imageryTest01_inplaneT2.nii.gz"))))

file = nib.load(filepath)

data = file.get_fdata()
print(data)

#fig, axes = plt.subplots(1,3, figsize=(15,35))
#axes = np.ravel(axes)

plt.imshow(data[:,:,36])#.T, cmap='gray', origin='lower') # Saggital
plt.show()#axes[0].set_title("Saggital")
plt.imshow(data[:,:,2])#.T, cmap='gray', origin='lower') # Saggital
plt.show()#axes[0].set_title("Saggital")

#plotting.plot_matrix(data)