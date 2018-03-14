import pandas as pd
import numpy as np
import scipy.misc

import sklearn.metrics as sk
from sklearn import svm
from sklearn import preprocessing

def normalizer(m):
	return preprocessing.normalize(m, axis = 1, norm = 'l2')

print('start')
x = np.loadtxt("./dataset/train_x.csv", delimiter=",") # load from text 
y = np.loadtxt("./dataset/train_y.csv", delimiter=",") 
x_test = np.loadtxt("./dataset/test_x.csv", delimiter=",") 
x = x.reshape(-1, 64, 64) # reshape 
y = y.reshape(-1, 1) 
x_test = x_test.reshape(-1, 64, 64)
norm_x = normalizer(x)
norm_x_test = normalizer(x_test)
print('data initialized')

#first 100

scipy.misc.imshow(x[0]) # to visualize only 

classifier = svm.LinearSVC()
classifier.fit(norm_x[:100],y[:100])
test_predict = classifier.predict(norm_x_test[:100])


score = sk.f1_score(norm_x_test[:100], test_predict, average='micro')
print('test' + str(score))