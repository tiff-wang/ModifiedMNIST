import pandas as pd
import numpy as np
import scipy.misc

import sklearn.metrics as sk
from sklearn import svm
from sklearn import preprocessing

# use a L2 normalizer for regularization purposes
def normalizer(m):
	return preprocessing.normalize(m, axis = 1, norm = 'l2')

print('parsing train_x')
x = np.loadtxt("../dataset/train_x_preproc.csv", delimiter=",") # load from text 

print('parsing train_y')
y = np.loadtxt("../dataset/train_y.csv", delimiter=",") 

print('parsing test_x')
x_test = np.loadtxt("../dataset/test_x_preproc.csv", delimiter=",") 

# reshape the dataset into
print('data initialized')

classifier = svm.LinearSVC()
classifier.fit(x,y)

test_predict = classifier.predict(x_test)

arr = np.arange(len(test_predict))

np.savetxt('svm_output.csv', np.dstack((arr, test_predict))[0], "%d,%d", header = "Id,Label", comments='')
