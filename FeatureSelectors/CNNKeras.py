import numpy as np

from keras.models import Sequential
from keras.models import Model
from keras.layers import add
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Reshape
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.utils import plot_model

N_FILTERS = 8
KERNEL_SIZE = 16
POOL_SIZE = 16
DENSE_SIZE = 100
N_EPOCHS = 100

def VGG(input_shape, n_filters, kernel_size, pool_size, dense_size, n_outputs):
    inp = Input(shape=input_shape)
    # start VGG module 1
    conv = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', activation='relu')(inp)
    mp = MaxPooling1D(pool_size=pool_size)(conv)
    flat = Reshape((mp.shape[1] * mp.shape[2], 1))(mp)

    # start VGG module 2
    conv = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', activation='relu')(flat)
    mp = MaxPooling1D(pool_size=pool_size)(conv)
    flat = Reshape((mp.shape[1] * mp.shape[2], 1))(mp)

    # start VGG module 3
    conv = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', activation='relu')(flat)
    mp = MaxPooling1D(pool_size=pool_size)(conv)
    flat = Flatten()(mp)

    # end VGG modules
    d1 = Dense(dense_size, activation='relu')(flat)
    d2 = Dense(n_outputs, activation='softmax')(d1)
    model = Model(inputs=inp, outputs=d2)
    model.summary()
    return model

def Inception(input_shape, n_filters, kernel_size, pool_size, dense_size, n_outputs):
    inp = Input(shape=input_shape)
    # start inception module 1
    conv1 = Conv1D(n_filters, 1, padding='same', activation='relu')(inp)

    conv3 = Conv1D(n_filters, 3, padding='same', activation='relu')(inp)

    conv5 = Conv1D(n_filters, 5, padding='same', activation='relu')(inp)

    pool = MaxPooling1D(pool_size=pool_size)(inp)
    pool = Conv1D(n_filters, 1, padding='same', activation='relu')(pool)

    concat = concatenate([conv1, conv3, conv5, pool], axis=1)
    #concat = Reshape((concat.shape[1] * concat.shape[2], 1))(concat)


    # start inception module 2
    conv1 = Conv1D(1, 1, padding='same', activation='relu')(concat)
    conv1 = Conv1D(n_filters, 1, padding='same', activation='relu')(conv1)

    conv3 = Conv1D(1, 1, padding='same', activation='relu')(concat)
    conv3 = Conv1D(n_filters, 3, padding='same', activation='relu')(conv3)

    conv5 = Conv1D(1, 1, padding='same', activation='relu')(concat)
    conv5 = Conv1D(n_filters, 5, padding='same', activation='relu')(conv5)

    pool = MaxPooling1D(pool_size=pool_size)(concat)
    pool = Conv1D(n_filters, 1, padding='same', activation='relu')(pool)

    concat = concatenate([conv1, conv3, conv5, pool], axis=1)
    #concat = Reshape((concat.shape[1] * concat.shape[2], 1))(concat)

    # start inception module 3
    '''
    conv1 = Conv1D(1, 1, padding='same', activation='relu')(concat)
    conv1 = Conv1D(n_filters, 1, padding='same', activation='relu')(conv1)

    conv3 = Conv1D(1, 1, padding='same', activation='relu')(concat)
    conv3 = Conv1D(n_filters, 3, padding='same', activation='relu')(conv3)

    conv5 = Conv1D(1, 1, padding='same', activation='relu')(concat)
    conv5 = Conv1D(n_filters, 5, padding='same', activation='relu')(conv5)

    pool = MaxPooling1D(pool_size=pool_size)(concat)
    pool = Conv1D(n_filters, 1, padding='same', activation='relu')(pool)

    concat = concatenate([conv1, conv3, conv5, pool], axis=1)
    concat = Conv1D(1, 1, padding='same', activation='relu')(concat)
    #concat = Reshape((concat.shape[1] * concat.shape[2], 1))(concat)
    '''

    # end inception modules
    concat = Reshape((concat.shape[1] * concat.shape[2], 1))(concat)
    flat = Flatten()(concat)
    d1 = Dense(dense_size, activation='relu')(flat)
    d2 = Dense(n_outputs, activation='softmax')(d1)
    model = Model(inputs=inp, outputs=d2)
    model.summary()

    return model

def ResNet(input_shape, n_filters, kernel_size, pool_size, dense_size, n_outputs):
    inp = Input(shape=input_shape)
    # start Res module 1
    conv1 = Conv1D(n_filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(inp)
    conv2 = Conv1D(n_filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    sum = add([conv2, inp])
    act = Activation('relu')(sum)
    # start Res module 2
    conv1 = Conv1D(n_filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(act)
    conv2 = Conv1D(n_filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    sum = add([conv2, act])
    act = Activation('relu')(sum)
    # start Res module 3
    conv1 = Conv1D(n_filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(act)
    conv2 = Conv1D(n_filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    sum = add([conv2, act])
    act = Activation('relu')(sum)
    # End Res module
    flat = Flatten()(act)
    d1 = Dense(dense_size, activation='relu')(flat)
    d2 = Dense(n_outputs, activation='softmax')(d1)
    model = Model(inputs=inp, outputs=d2)
    model.summary()
    return model

class CNNKeras:

    def __init__(self, model_name, n_features, n_outputs, n_filters = N_FILTERS, kernel_size = KERNEL_SIZE, pool_size = POOL_SIZE, dense_size = DENSE_SIZE):
        '''
        self.model = Sequential()
        self.model.add(Input(shape=(n_features, 1)))
        #self.model.add(Dense(n_features, activation='relu'))
        self.model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(dense_size, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        '''
        '''
        inp = Input(shape=(n_features, 1))
        conv = Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu')(inp)
        mp = MaxPooling1D(pool_size=pool_size)(conv)
        flat = Flatten()(mp)
        d1 = Dense(dense_size, activation='relu')(flat)
        d2 = Dense(n_outputs, activation='softmax')(d1)
        self.model = Model(inputs=inp, outputs=d2)
        '''

        self.model_name = model_name

        if model_name == 'VGG':
            self.model = VGG(input_shape=(n_features, 1), n_filters=n_filters, kernel_size=kernel_size, pool_size=pool_size, dense_size=dense_size, n_outputs=n_outputs)
        elif model_name == 'INC':
            self.model = Inception(input_shape=(n_features, 1), n_filters=n_filters, kernel_size=kernel_size, pool_size=pool_size, dense_size=dense_size, n_outputs=n_outputs)
        elif model_name == 'RES':
            self.model = ResNet(input_shape=(n_features, 1), n_filters=n_filters, kernel_size=kernel_size, pool_size=pool_size, dense_size=dense_size, n_outputs=n_outputs)
        else:
            raise Exception('Invalid model name'+model_name)
        #self.model.summary()
        #plot_model(self.model)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        #print(self.model.summary())

    def fit(self, x, y):
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        y = to_categorical(y)
        if self.model_name == 'INC':
            print("start fitting...")
        self.model.fit(x, y, verbose=0, epochs=N_EPOCHS)
        if self.model_name == 'INC':
            print("done fitting.")

    def score(self, x, y):
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        y = to_categorical(y)
        if self.model_name == 'INC':
            print("start evaluating...")
        _, accuracy = self.model.evaluate(x, y, verbose=0)
        if self.model_name == 'INC':
            print("done evaluating.")

        #print("predicting...")
        #print(np.argmax(self.model.predict(x), axis=1))
        #print(np.argmax(y, axis=1))
        return accuracy
