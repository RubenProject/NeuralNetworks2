'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
from __future__ import print_function

import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

#Safeguard
if os.environ['CUDA_VISIBLE_DEVICES'] == '':
    print('Specify which GPUs are visible!!')
    exit()

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#shuffle inputs
def shuffle_inputs():
    order = np.random.permutation(list(range(0, 784)))
    for i in range(0, len(x_train)):
        x_train[i] = x_train[i][order]

    for i in range(0, len(x_test)):
        x_test[i] = x_test[i][order]


def create_mlp():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
    return model

def evaluate(model):
    predictions = model.predict_classes(x_test, verbose=1)
    y_test_classes = np.array(list(map(np.argmax, y_test)))
    
    cm = confusion_matrix(y_test_classes, predictions)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3)

    return cm



shuffle_inputs()
mlp = create_mlp()
res = np.array([evaluate(mlp) for i in range(0, 100)])

print(res.mean(axis=0))
#Shows non controllable randomness
print(res.std(axis=0))

