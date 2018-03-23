# Team Naive Baes 
## COMP 551 - Applied Machine Learning Project 4

Frederic Ladouceur 2606
Joshua Lau 260582760
Tiffany Wang 260684152

### HOW TO RUN OUR CODE

This is a guideline to where to find the code for each part, and how to run it. 


##### Preprocess 
Directory: . /preprocess

This code is in a jupyter notebook format. 
The preprocessing file will generate new data csv files. 
We did not include the output files in the project submission because they are too large. 

However you can find them in the following links: 
- train_x: 'https://s3.us-east-2.amazonaws.com/kaggle551/train_x_preproc.csv'
- train_y: 'https://s3.us-east-2.amazonaws.com/kaggle551/train_y.csv'
- test_x: 'https://s3.us-east-2.amazonaws.com/kaggle551/test_x_preproc.csv'


##### 1) Basic Linear Learner - SVM 
Directory: ./svm 

Run the code using the following command inside the svm folder: 
```
python svm.py 
```

The output file with the test prediction with be saved within the folder under the name 'svm_output.csv'


##### 2) Neural Net
Directory: ./NN

This code is in a jupyter notebook format. 

The output file with the test prediction with be saved within the folder under the name 'nn_output.csv'


##### 3) Kaggle Competition models 
Directory: ./competition

This folder contains several subfolders with different models. In each folder, you can run the model using: 
```
python kaggle_data_aug.py 
```

The output file will be under the name 'prediction_output_{}.csv' where {} contains the validation accuracy.


##### CSV parsing 
Directory: .
The output of the models is a (10000, 10) matrix where each vector hold a probabilistic metrics for each class. The CSV parsing file will parse this (10000, 10) into the submission format for the kaggle competition. 

Run the script from the root folder using 
```
python csv_parser.py -f 'filename'
``` 

The submission file will be generated in the same folder as the parsed file. 


##### Ensemble 
Directory: ./Ensemble
In order to get better accuracy, we used the ensembling method. The 'outputs' folder contains prediction files with the best accuracies. Running the 'ensemble.py' script will generate an output file 'ensemble_output.csv' ready for submission! 

Run using 
```
python ensemble.py
```

A jupyter notebook version of this code is also available in the same folder. 
