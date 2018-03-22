import pandas as pd
import numpy as np
import scipy.misc

import sklearn.metrics as sk
from sklearn import svm
from sklearn import preprocessing

validation_split = 0.05

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

# split validation and training data 
split = len(x_train) * validation_split
x_valid = x_train[:split]
y_valid = y_train[:split]

x_train = x_train[split:]
y_train = y_train[split:]

# reshape the dataset into
print('data initialized')
max_iter = [250 * i for i in range(10)]
multi_class = ['ovr', 'crammer_singer']
penalty = ['l1', 'l2']


param = [{'max_iter': max_iter, 'multi_class': multi_class, 'penalty': penalty}]

combine_input = sparse.vstack([x_train, x_valid])
combine_truth = np.concatenate((y_train, y_valid))
fold = [-1 for i in range(x_train.shape[0])] + [0 for i in range(x_valid.shape[0])]
clf = svm.LinearSVC()
ps = PredefinedSplit(test_fold = fold)
clf = GridSearchCV(clf, param, cv=ps, refit=True)
clf.fit(combine_input, combine_truth)

best_param = clf.best_params_

f1_train = f1_score(y_train, clf.predict(x_train), average = average)
f1_valid = f1_score(y_valid, clf.predict(x_valid), average = average)

test_predict = clf.predict(x_test)

arr = np.arange(len(test_predict))

np.savetxt('svm_output.csv', np.dstack((arr, test_predict))[0], "%d,%d", header = "Id,Label", comments='')
