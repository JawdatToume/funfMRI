import os
import re
import numpy as np
list_of_files=os.listdir("/mnt/288fa9dd-e9f7-45ac-8dc8-5f5ddeb362e4/fmriTDA_lin_reg_on_some_data/subject_wise_numpy_arrays/")

X_files=list()
Y_files=list()
for file_name in list_of_files:
	
	if re.match(r".+X_data.+",file_name):
		X_files.append(file_name)
	if re.match(r".+Y_data.+",file_name):
		Y_files.append(file_name)

X_files=sorted(X_files)
Y_files=sorted(Y_files)
	
data_x=np.load("/mnt/288fa9dd-e9f7-45ac-8dc8-5f5ddeb362e4/fmriTDA_lin_reg_on_some_data/subject_wise_numpy_arrays/"+X_files[0])
data_y=np.load("/mnt/288fa9dd-e9f7-45ac-8dc8-5f5ddeb362e4/fmriTDA_lin_reg_on_some_data/subject_wise_numpy_arrays/"+Y_files[0])

print(data_x.shape)
print(data_y.shape)

for layer in range(0,7):
	x=data_x[layer]
	y=data_y[layer]
	print(x.shape)
	print(y.shape)


