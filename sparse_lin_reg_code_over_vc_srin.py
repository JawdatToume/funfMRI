import os
import re
import numpy as np
import slir

list_of_files=os.listdir("/home/srinjoy/PycharmProjects/funfMRI/subject_wise_numpy_arrays/")

X_files=list()
Y_files=list()
for file_name in list_of_files:
	
	if re.match(r".+test_X.+",file_name):
		X_files.append(file_name)
	if re.match(r".+test_Y.+",file_name):
		Y_files.append(file_name)

X_files=sorted(X_files)
Y_files=sorted(Y_files)


data_x=np.squeeze(np.load("/home/srinjoy/PycharmProjects/funfMRI/subject_wise_numpy_arrays/"+X_files[0]),0)
data_y=np.squeeze(np.load("/home/srinjoy/PycharmProjects/funfMRI/subject_wise_numpy_arrays/"+Y_files[0]),0)

# print(data_x.shape)
# print(data_y.shape)
lin_regs=list()
for layer in range(data_x.shape[0]):

	x=data_x[layer,:]
	y=data_y[layer,:]
	model=slir.SparseLinearRegression(n_iter=10 ,verbose=True)
	print(x.shape)
	print(y.shape)
	slr=model.fit(x,y)
	lin_regs.append(slr)
	predicted_features = slr.predict(x)
	corrmat = np.corrcoef(predicted_features, y)

	print("Correalation: %.4f" % corrmat[0, 1])
	print("MSE: %.4f" % np.mean((predicted_features - y) ** 2))

# TODO add useage for looping over multiple images and fiting the model
# TODO fix the other two make_data python scripts which save the array properly