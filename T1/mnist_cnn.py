'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

#Safeguard
if os.environ['CUDA_VISIBLE_DEVICES'] == '':
    print('Specify which GPUs are visible!!')
    exit()

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#shuffle inputs
def shuffle_inputs(x, o):
    x = x.reshape(len(x), 784)
    for i in range(0, len(x)):
        x[i] = x[i][o]
    x = x.reshape(len(x), 28, 28)
    return x

order = np.random.permutation(list(range(0, 784)))
x_train = shuffle_inputs(x_train, order)
x_test = shuffle_inputs(x_test, order)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def create_cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
    return model

def test(model):
    predictions = model.predict_classes(x_test, verbose=0)
    y_test_classes = np.array(list(map(np.argmax, y_test)))

    cm = confusion_matrix(y_test_classes, predictions)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3)

    return cm

test_acc = 0
train_acc = 0
for i in range(0, 10):
    cnn = create_cnn()
    score = cnn.evaluate(x_train, y_train, verbose=0)
    train_acc += score[1]
    score = cnn.evaluate(x_test, y_test, verbose=0)
    test_acc += score[1]

print('Train accuracy:', train_acc/10)
print('Test accuracy:', test_acc/10)


print('Generating CM...')
res = np.array([test(cnn) for i in range(0, 100)])

print(res.mean(axis=0))
print(np.max(res.std(axis=0)))

