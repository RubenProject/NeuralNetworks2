from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

#Safegaurd
if os.environ['CUDA_VISIBLE_DEVICES'] == '':
    print('Environment not set!')
    exit()


MAX_BIT = 32


def encode(x):
    return list(reversed(list(map(float, "{0:b}".format(x)))))


def decode(x):
    return int(''.join(str(int(round(i))) for i in reversed(x)), 2)


def gen_entry():
    a = np.random.randint(2 ** MAX_BIT)
    b = np.random.randint(2 ** MAX_BIT)
    c = a + b
    [a, b, c] = encode(a), encode(b), encode(c)
    [a, b, c] = pad_sequences([a, b, c], padding='post', dtype='float32', maxlen=MAX_BIT+1)
    x = np.array(list(zip(a, b)))
    return x, c


def gen_data(N):
    x = np.zeros((N, MAX_BIT + 1, 2))
    y = np.zeros((N, MAX_BIT + 1, 1))
    for i in range(N):
        xi, yi = gen_entry()
        x[i, :, :] = xi
        y[i, :, :] = yi.reshape(1, -1, 1)
    x = x.reshape(N, -1, 2)
    y = y.reshape(N, -1, 1)
    return x, y


def create_rnn(N=10000, epochs=40):
    model = Sequential()
    model.add(SimpleRNN(4, input_shape=(None, 2), return_sequences=True))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    [x, y] = gen_data(N)
    model.fit(x, y, batch_size=40, verbose=1, epochs=epochs)
    return model


def test_rnn(model, N=1000):
    [x, y] = gen_data(N)
    score = model.evaluate(x, y, verbose=0)
    print("test loss: ", score[0])
    print("test accuracy: ", score[1])
    for i in range(10):
        a = np.random.randint(2 ** MAX_BIT)
        b = np.random.randint(2 ** MAX_BIT)
        c = a + b
        c_pred = add_rnn(model, a, b)
        if c == c_pred:
            print("%i + %i = %i ☑ " % (a, b, c_pred))
        else:
            print("%i + %i = %i ☒ " % (a, b, c_pred))



def add_rnn(model, a, b):
    [a, b] = encode(a), encode(b)
    [a, b] = pad_sequences([a, b], padding='post', dtype='float32', maxlen=None)
    x = np.dstack((a, b)).reshape(1, -1, 2)
    y = model.predict(x).ravel()
    y = decode(y)
    return y
    

rnn = create_rnn()
#rnn = load_model('bin_add.h5')
rnn.save('bin_add_4.h5')
test_rnn(rnn)

