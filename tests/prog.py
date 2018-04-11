#---------------------------------------------------------------
#Set environment

import os

os.environ['CUDA_VISIBLE_DEVICES']= '5'

#---------------------------------------------------------------
#imports

from keras.models import  Sequential 
from keras.layers import Dense, Dropout, Activation 
from keras.optimizers import  SGD 

import csv
import numpy as np

#---------------------------------------------------------------
#load data

def open_csv(fname):
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        l = list(f)
        l = [i.split(',') for i in l]
    t = []
    for l in l:
        t.append(np.array([float(i) for i in l]))
    l = t
    return np.array(l)


x_train = open_csv('data/train_in.csv')
y_train = open_csv('data/train_out.csv')

t = np.zeros((len(y_train), 10), dtype=float)
for i, j in enumerate(y_train):
    t[i][int(j)] = 1.0
y_train = t


x_test = open_csv('data/test_in.csv')
y_test = open_csv('data/test_out.csv')

t = np.zeros((len(y_test), 10), dtype=float)
for i, j in enumerate(y_test):
    t[i][int(j)] = 1
y_test = t

#---------------------------------------------------------------
#analysis

model = Sequential() 

#256 input vector
#64 hidden nodes
#10 output nodes
model.add(Dense(64, activation='relu', input_dim=256)) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(10, activation='softmax')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#train
model.fit(x_train, y_train, epochs=20, batch_size=128)

#test
score = model.evaluate(x_test, y_test, batch_size=128)

print(score)
