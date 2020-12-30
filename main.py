from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Reshape
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import UpSampling2D, Conv2D
from tensorflow.python.keras.layers import ELU
from tensorflow.python.keras.layers import Flatten, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.datasets import mnist

import os
from PIL import Image
from helper import *

def generator(input_dim=100, units=1024, activation='relu'):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=units))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Reshape((7,7,128),input_shape=(128*7*7,)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64, (5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (5,5), padding='same'))
    model.add(Activation('tanh'))
    print(model.summary())
    return model

def discriminator(input_shape=(28,28,1), nb_filter=64):
    model = Sequential()
    model.add(Conv2D(nb_filter, (5,5), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(2*nb_filter, (5,5), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(4*nb_filter))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model




















