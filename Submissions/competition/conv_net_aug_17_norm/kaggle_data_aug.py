from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

def norm_input(x):
	return (x - x.mean().astype(np.float32)/x.std().astype(np.float32))


# batch-size: number of samples that going to propagate through the network 
batch_size = 64
num_classes = 10
epochs = 17
validation_split = 0.05

# input image dimensions
img_rows, img_cols = 64, 64

URL = 'https://s3.us-east-2.amazonaws.com/kaggle551/'

# load the data
#x_train = np.loadtxt('../dataset/train_x_proc.csv', delimiter = ',')
#y_train = np.loadtxt('../dataset/train_y.csv', delimiter = ',')[:-1]

x_train = pd.read_csv(URL + 'train_x_preproc.csv', header=None)
y_train = pd.read_csv(URL + 'train_y.csv', header=None)

x_train = np.array(x_train.as_matrix())
y_train = np.array(y_train[0])

#x_test = np.loadtxt('../dataset/test_x_proc.csv', delimiter = ',')
x_test = pd.read_csv(URL + 'test_x_preproc.csv', header=None)
x_test = np.array(x_test.as_matrix())

# split train /validation
split = int(x_train.shape[0] * 0.05)
x_valid = x_train[:split]
y_valid = y_train[:split]

x_train = x_train[split:]
y_train = y_train[split:]


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

mean_x = x_train.mean().astype(np.float32)
std_x = x_train.std().astype(np.float32)

def norm_input(x): return (x-mean_x)/std_x

if K.image_data_format() == 'channels_first':
    train_reshaped = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    test_reshaped = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    valid_reshaped = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_reshaped = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    test_reshaped = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    valid_reshaped = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


model = Sequential()
model.add(Lambda(norm_input, input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.3,
                         height_shift_range=0.1, zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(train_reshaped, y_train, batch_size=batch_size)
test_generator = test_gen.flow(valid_reshaped, y_valid, batch_size=batch_size)



# verbose = 1 // log output
model.fit_generator(train_generator, steps_per_epoch=train_reshaped.shape[0]//batch_size, epochs=epochs, 
                    validation_data=test_generator, validation_steps=valid_reshaped.shape[0]//batch_size)

score = model.evaluate(valid_reshaped, y_valid, verbose=0)
score = (score[1] * 100)
test_predict = model.predict(test_reshaped, verbose=1)
test_predict = np.argmax(test_predict, axis=1)
arr = np.arange(len(test_predict))
np.savetxt('predict_output_{}.csv'.format(int(score)), np.dstack((arr, test_predict))[0], "%d,%d", header = "Id,Label", comments='')
