import os
import pickle
import torch
import slir
import numpy as np
main_dir="/home/srinjoy/PycharmProjects/funfMRI/sub_wise_lin_reg_data/"
main_data_files=os.listdir(main_dir)

for data_file_pth in main_data_files:
    print("For file :",data_file_pth)
    with open(main_dir+data_file_pth, 'rb')as data_file:
        list_of_data= pickle.load(data_file)
        per_subject_x_data=None
        per_subject_y_data = None
        if len(list_of_data) != 0:
            #print("Number of samples in the file:",len(list_of_data))
            for index,data_samples in enumerate(list_of_data):

                fmri_1d_vector=data_samples[0]
                dict_of_multi_layer_features=data_samples[1]
                img_used=data_samples[2]
                print("\t Index : ", index, "running with Img_ID:",img_used)
                #print(fmri_1d_vector.shape)

                per_data_sample_x_data=None
                per_data_sample_y_data = None
                for layer_names in dict_of_multi_layer_features.keys():
                    per_layer_sample_x_data=None
                    per_layer_sample_y_data=None

                    for feature_value in dict_of_multi_layer_features[layer_names]:

                        #print(fmri_1d_vector.shape)
                        feature_value=np.expand_dims(feature_value.detach().numpy(),axis=0)
                        #print(feature_value.shape)

                        if per_layer_sample_x_data is None and per_layer_sample_y_data is None:
                            #print("Init the arrays")
                            per_layer_sample_x_data=fmri_1d_vector
                            per_layer_sample_y_data=feature_value

                        else:

                            per_layer_sample_x_data=np.concatenate((per_layer_sample_x_data,fmri_1d_vector),axis=0)
                            per_layer_sample_y_data=np.concatenate((per_layer_sample_y_data,feature_value),axis=0)

                        # training_x_y_per_features_value=(fmri_1d_vector,feature_value)
                        #
                    # print(per_layer_sample_x_data.shape)
                    # print(per_layer_sample_y_data.shape)
                    #print(len(per_layer_sample_x_y_data))
                    per_layer_sample_x_data = np.expand_dims(per_layer_sample_x_data, 0)
                    per_layer_sample_y_data = np.expand_dims(per_layer_sample_y_data, 0)
                    #merge over all the layers
                    if per_data_sample_x_data is None and per_data_sample_y_data is None:

                        per_data_sample_x_data=per_layer_sample_x_data
                        per_data_sample_y_data=per_layer_sample_y_data

                    else:
                        per_data_sample_x_data = np.concatenate((per_data_sample_x_data,per_layer_sample_x_data),axis=0)
                        per_data_sample_y_data = np.concatenate((per_data_sample_y_data,per_layer_sample_y_data),axis=0)




                per_data_sample_x_data = np.expand_dims(per_data_sample_x_data, 0)
                per_data_sample_y_data = np.expand_dims(per_data_sample_y_data, 0)

                # merge over all the layers
                if per_subject_x_data is None and per_subject_y_data is None:

                    per_subject_x_data = per_data_sample_x_data
                    per_subject_y_data = per_data_sample_y_data


                else:

                    per_subject_x_data = np.concatenate((per_subject_x_data, per_data_sample_x_data), axis=0)
                    per_subject_y_data = np.concatenate((per_subject_y_data, per_data_sample_y_data), axis=0)

                print(f"\t \t \t ./subject_wise_numpy_arrays/{data_file_pth[:-4]}_X_data_imgused_{img_used}_data_index_{index}.npy")
                print(f"\t \t \t \t X data array_done for file saved shape :", per_subject_x_data.shape)
                print(f"\t \t \t./subject_wise_numpy_arrays/{data_file_pth[:-4]}_Y_data_imgused_{img_used}_data_index_{index}.npy")
                print(f"\t \t \t \t Y data array_done for file saved shape :", per_subject_y_data.shape)

                with open(f"./subject_wise_numpy_arrays/{data_file_pth[:-4]}_X_data_imgused_{img_used}_data_index_{index}.npy","wb") as f_numpy_X_data:
                    np.save(f_numpy_X_data,per_data_sample_x_data)
                with open(f"./subject_wise_numpy_arrays/{data_file_pth[:-4]}_Y_data_imgused_{img_used}_data_index_{index}.npy","wb") as f_numpy_Y_data:
                    np.save(f_numpy_Y_data,per_data_sample_y_data)
            # TODO Fix the last part of this code it has a memory overload before i can run it
            # THE SAVE ORDER IS SUBJECT_[TRAIN_OR_TEST].NP INSIDE IS SUBJECT NUMBER * LAYERS OF CNN X NUMBERS OF FEATURE VALUES PREDICTED X FMRI_VECTOR_SIZE

        print(f"X data array_done for file {data_file_pth}:",per_subject_x_data.shape)
        print(f"Y data array_done for file {data_file_pth}:",per_subject_y_data.shape)













