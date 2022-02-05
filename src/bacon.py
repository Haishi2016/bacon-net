from abc import abstractmethod


from abc import ABC
import tensorflow as tf
from tensorflow import keras
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import numpy as np

# surpress warnings on running GPU Tensorflow on CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class baconNet(ABC):
    def __custom_activation(self, x):
        return x

    def __init__(self, size, constantTerm=True):
        get_custom_objects().update(
            {'custom_activation': Activation(self.__custom_activation)})
        self.__model = tf.keras.models.Sequential()
        self.__model.add(tf.keras.Input(shape=(size,)))
        self.__model.add(tf.keras.layers.Dense(
            size, activation=self.__custom_activation))
        self.__model.add(tf.keras.layers.Dense(
            1, activation=self.__custom_activation))
        self.__model.compile(
            optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        self.__size = size
        self.__constantTerm = constantTerm

    @abstractmethod
    def explain(self, singleVariable=False):
        pass

    @abstractmethod
    def get_contribution(self, a, b):
        m = np.matmul(
            self.__model.layers[0].weights[0], self.__model.layers[1].weights[0])
        if self.__constantTerm:
            # assuming the last input is a constant column
            bias0 = self.__model.layers[0].bias.numpy()
            weight0 = self.__model.layers[0].weights[0][self.__size-1]
            for i in range(len(bias0)):
                bias0[i] = bias0[i] + weight0[i]
            weight1 = self.__model.layers[1].weights[0]
            for i in range(len(bias0)):
                bias0[i] = bias0[i] * weight1[i][0]
            c = bias0.sum() + self.__model.layers[1].bias.numpy()[0]
        else:
            c = 0
        return m, c

    @abstractmethod
    def expand(self, a, b, y):
        pass

    def predict(self, a, b=0):
        simple = False
        if isinstance(a, list):
            if isinstance(b, list):
                x, X, y = self.expand(a, b, np.zeros(len(a)))
            else:
                x, X, y = self.expand(a, np.zeros(len(a)), np.zeros(len(a)))
        else:
            simple = True
            x, X, y = self.expand([a], [b], [0])
        if simple:
            return self.__model.predict(X)[0][0]
        else:
            return self.__model.predict(X).flatten()

    def fit(self, a, b, y):
        callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=20)
        x, X, y = self.expand(a, b, y)
        history = self.__model.fit(
            X, y, epochs=600, batch_size=10, verbose=0, callbacks=[callback])
        return history


class dataCreator:
    def create(size, scale, aggregate, singleVariable=False):
        a = np.random.rand(size) * scale
        if not singleVariable:
            b = np.random.rand(size) * scale
        else:
            b = np.zeros(size)
        y = aggregate(a, b)
        return a, b, y
