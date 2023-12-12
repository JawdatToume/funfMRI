# funfMRI

This is the code we used to generate the results by testing and training our models for our CMPUT 624 project.

We have code used for 3 different model-building techniques, the TDA, baseline, and our CNN.

Firstly, we have code used to collect our fMRI data, to extract it, load it, mask it, and prepare it to be used by our other files of code to perform our training and evaluation. loader.py is used for loading, masker.py is used for masking. These files are mostly intermediary files called by other functions.

In addition, we have regresser.py, which is used by each of the models to develop a linear regression model, this file generates the Sparse Linear Regression models, as well as labeler.py, which loads and labels our feature data for use in training and evaluation.

Our baseline is generated through roiaverager.py, typically, the main() function is adjusted to alter the different tests that are done or models that are trained for the baseline, each training run generates a unique pkl file to save time on training in the future, and then testing is handled within the file as well, restoring the pkl file if it already exists. Lines 38 (for training) and 42 (for testing) are of special interest, as those function calls determine which subject we are testing on, what layer we are checking for, and what particular experiments we are focusing on.


Each of the tests creates a csv which is used to generate barplot figures to show our results. plotgraphs.py is a file we use to generate these graphs, using plotgraphs.py [file] creates a barplot out of the data in [file].

## AlexNet and ConvNext-tiny image feature prediction

Preprocessed Image Features generated by us can be downloaded from:  [Img Features Drive Link](https://drive.google.com/drive/folders/1u3ZibkBIougeTN30zn0_p1VGc_S0RkLi?usp=sharing)

The Image features were generated using the [notebooks](https://github.com/JawdatToume/funfMRI/tree/main/python_notebooks) in the python_notebooks folder in the repository.

The code in [Modified Kamitani Lab Generic Object Decoding](https://github.com/JawdatToume/funfMRI/blob/main/kamitani%20generic%20obj%20decoding_code_modified_for_convnext.zip) has been for most of the image feature prediction training and analysis of the AlexNet Image Features regenerated and ConvNext-tiny Image features.







