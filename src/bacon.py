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
        self.__model.add(tf.keras.Input(shape=(2,)))
        self.__model.add(expansionLayer(self.expand))
        self.__model.add(tf.keras.layers.Dense(
            size, activation=self.__custom_activation))
        self.__model.add(tf.keras.layers.Dense(
            1, activation=self.__custom_activation))
        self.__model.compile(
            optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        self.__constantTerm = constantTerm
        self.__size = size

    @ abstractmethod
    def explain(self, singleVariable=False):
        pass

    @ abstractmethod
    def get_contribution(self, a, b):
        m = np.matmul(
            self.__model.layers[1].weights[0], self.__model.layers[2].weights[0])
        if self.__constantTerm:
            # assuming the last input is a constant column
            bias0 = self.__model.layers[1].bias.numpy()
            weight0 = self.__model.layers[1].weights[0][self.__size-1]
            for i in range(len(bias0)):
                bias0[i] = bias0[i] + weight0[i]
            weight1 = self.__model.layers[2].weights[0]
            for i in range(len(bias0)):
                bias0[i] = bias0[i] * weight1[i][0]
            c = bias0.sum() + self.__model.layers[2].bias.numpy()[0]
        else:
            c = 0
        return m, c

    @ abstractmethod
    def expand(self, a, b):
        pass

    def predict(self, a, b=0):
        simple = False
        if isinstance(a, list):
            if isinstance(b, list):
                x = np.column_stack((a, b))
            else:
                x = np.column_stack((a, np.zeros(len(a))))
        else:
            simple = True
            x = np.column_stack(([a], [b]))
        if simple:
            return self.__model.predict(x)[0][0]
        else:
            return self.__model.predict(x).flatten()

    def fit(self, a, b, y):
        callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=20)
        history = self.__model.fit(
            np.column_stack((a, b)), y, epochs=600, batch_size=10, verbose=0, callbacks=[callback])
        return history

    def get_model(self):
        return self.__model


class dataCreator:
    def create(size, scale, aggregate, params=2):
        values = []
        for i in range(params):
            values.append(np.random.rand(size) * scale)
        if len(values) < 2:
            values.append(np.zeros(size))
        y = aggregate(values)
        return values, y


class expansionLayer(tf.keras.layers.Layer):
    def __init__(self, expander, **kwargs):
        super().__init__(**kwargs)
        self.expander = expander

    def call(self, inputs):
        X = self.expander(
            inputs[:, 0:1], inputs[:, 1:2])
        return X[0]


class expression:
    def __init__(self, terms):
        self.terms = terms

    def string(self, precision):
        strs = []
        for t in self.terms:
            strs.append(t.string(precision))
        return " + ".join(filter(None, strs))


class term:
    def __init__(self, coefficient=0, leftExp=None, rightExp=None, leftOpt="", rightOpt=""):
        self.leftExp = leftExp
        self.rightExp = rightExp
        self.leftOpt = leftOpt
        self.rightOpt = rightOpt
        self.coefficient = coefficient

    def string(self, precision=-1):
        if self.coefficient == 0:
            return ""
        if precision == -1:
            ret = str(self.coefficient)
        else:
            ret = str(round(self.coefficient, precision))
        if self.leftExp != None:
            if isinstance(self.leftExp, expression):
                ret += "(" + self.leftExp.string() + ")" + self.leftOpt
            else:
                ret += self.leftExp + self.leftOpt
        if self.rightExp != None:
            if isinstance(self.rightExp, expression):
                ret += "(" + self.rightExp.string() + ")" + self.rightOpt
            else:
                ret += self.rightExp + self.rightOpt
        return ret
