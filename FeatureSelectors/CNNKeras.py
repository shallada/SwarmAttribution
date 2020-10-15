import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

N_FILTERS = 8
KERNEL_SIZE = 16
POOL_SIZE = 16
DENSE_SIZE = 100
N_EPOCHS = 100

class CNNKeras:

    def __init__(self, n_features, n_outputs, n_filters = N_FILTERS, kernel_size = KERNEL_SIZE, pool_size = POOL_SIZE, dense_size = DENSE_SIZE):
        self.model = Sequential()
        self.model.add(Input(shape=(n_features, 1)))
        #self.model.add(Dense(n_features, activation='relu'))
        self.model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(dense_size, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #print(self.model.summary())

    def fit(self, x, y):
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        y = to_categorical(y)
        self.model.fit(x, y, verbose=0, epochs=N_EPOCHS)

    def score(self, x, y):
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        y = to_categorical(y)
        _, accuracy = self.model.evaluate(x, y, verbose=0)
        #print("predicting...")
        #print(np.argmax(self.model.predict(x), axis=1))
        #print(np.argmax(y, axis=1))
        return accuracy
