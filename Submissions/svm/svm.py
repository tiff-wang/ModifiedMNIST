import pandas as pd
import numpy as np
import scipy.misc

import sklearn.metrics as sk
from sklearn import svm
from sklearn import preprocessing

# use a L2 normalizer for regularization purposes
def normalizer(m):
	return preprocessing.normalize(m, axis = 1, norm = 'l2')

URL = 'https://s3.us-east-2.amazonaws.com/kaggle551/'

# load the data
x_train = pd.read_csv(URL + 'train_x_preproc.csv', header=None)
y_train = pd.read_csv(URL + 'train_y.csv', header=None)

x_train = np.array(x_train.as_matrix())
y_train = np.array(y_train[0])

#x_test = np.loadtxt('../dataset/test_x_proc.csv', delimiter = ',')
x_test = pd.read_csv(URL + 'test_x_preproc.csv', header=None)
x_test = np.array(x_test.as_matrix())

# reshape the dataset into
print('data initialized')

classifier = svm.LinearSVC()
classifier.fit(x,y)

test_predict = classifier.predict(x_test)

arr = np.arange(len(test_predict))

np.savetxt('svm_output.csv', np.dstack((arr, test_predict))[0], "%d,%d", header = "Id,Label", comments='')
