import pandas as pd
import numpy as np
import scipy.misc

import sklearn.metrics as sk
from sklearn import svm
from sklearn import preprocessing

def normalizer(m):
	return preprocessing.normalize(m, axis = 1, norm = 'l2')

print('parsing train_x')
x = np.loadtxt("./dataset/train_x.csv", delimiter=",") # load from text 

print('parsing train_y')
y = np.loadtxt("./dataset/train_y.csv", delimiter=",") 

print('parsing test_x')
x_test = np.loadtxt("./dataset/test_x.csv", delimiter=",") 

x = x.reshape(-1, 64, 64) # reshape 
y = y.reshape(-1, 1) 
x_test = x_test.reshape(-1, 64, 64)
#norm_x = normalizer(x)
#norm_x_test = normalizer(x_test)
print('data initialized')

#first 100
x = x.reshape(-1, 4096)
classifier = svm.LinearSVC()
classifier.fit(x,y)

x_test = x_test.reshape(-1, 4096)
test_predict = classifier.predict(x_test)

arr = np.arange(len(test_predict))

np.savetxt('predict_output.csv', np.dstack((arr, test_predict))[0], "%d,%d", header = "Id,Label", comments='')
