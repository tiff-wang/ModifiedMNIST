# preprocess.ipynb

## Description
The goal of this preprocess file is to read data from the training and testing modified MNIST datasets and modify their examples in such a way that it useless features are ignored and important features are kept without introducing errors to the dataset. The algorithm performs 5 tasks:
1. Load the datasets (stored as csv files) 
2. Set the background of the images to zero 
3. Detect the digits and create bounding boxes around them
4. Delete the image that have a size smaller than 80% of the largest digit in the image
5. Save these new images into csv files

## Input
The algorithm takes as input the provided training and testing MNIST datasets stored as csv files.

## Output
The algorithm produces 2 csv files respectively containing the preprocessed training and testing data.

## Parameters
### BACK_TRESH
This parameter corresponds to the treshold at which a pixel is considered to be part of the background or a digit. Default value=235.
### MAX_LEN
This parameter corresponds to the maximum dimension that a bounding box can to be considered as the largest digit. When a bounding box is too large, it is possible that it is composed of 2 digits close to one another. Default value=30.
### DISCARD_PERC
This parameter corresponds to the minimum percentage of the largest bounding boxe's side that a digit that a digit height or width must reach in order to be kept in the image, i.e. if max(w_digit,h_digit)>=max(w_larges_digit,h_largest_digit), the digit is kept in the image, otherwise, it is deleted. Default value=0.8.



