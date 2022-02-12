from abc import abstractmethod


from abc import ABC
import tensorflow as tf
from tensorflow import keras
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import numpy as np
from sympy import *
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
    def explain_contribution(self, m, c, singleVariable=False):
        pass

    def explain(self, singleVariable=False):
        return self.explain_model(self.__model, singleVariable)

    def explain_model(self, model, singleVariable=False):
        m, c = self.get_contribution_from_model(
            self.__model, self.__size, self.__constantTerm)
        return self.explain_contribution(m, c, singleVariable)

    def get_contribution_from_model(self, model, size, constantTerm=True):
        m = np.matmul(model.layers[1].weights[0], model.layers[2].weights[0])
        if constantTerm:
            # assuming the last input is a constant column
            bias0 = model.layers[1].bias.numpy()
            weight0 = model.layers[1].weights[0][size-1]
            for i in range(len(bias0)):
                bias0[i] = bias0[i] + weight0[i]
            weight1 = model.layers[2].weights[0]
            for i in range(len(bias0)):
                bias0[i] = bias0[i] * weight1[i][0]
            c = bias0.sum() + model.layers[2].bias.numpy()[0]
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

    def string(self, precision=-1, ignoreOne=False, delta=0.01, namePattern='x', patternIndex=0, sympy=True):
        strs = []
        for t in self.terms:
            strs.append(t.string(precision, ignoreOne,
                        delta, namePattern, patternIndex, sympy))
        ret = " + ".join(filter(None, strs))
        return ret

    def simplify(self, exp, precision=0, ignoreOne=True):
        ex1 = simplify(exp.replace("+ -", "-"))
        ex1 = expand(ex1)
        ex2 = ex1
        if precision >= 0:
            for a in preorder_traversal(ex1):
                if isinstance(a, Float):
                    ex2 = ex2.subs(a, round(a, precision))
        strExp = str(simplify(ex2))
        strExp = strExp.replace('**', '^')
        if precision >= 0 and ignoreOne:
            strExp = strExp.replace(str(round(1.0, precision))+'*', '')
        strExp = strExp.replace('*', '')
        return strExp


class term:
    def __init__(self, coefficient=0, term="", leftExp=None):
        self.coefficient = coefficient
        self.term = term
        self.leftExp = leftExp

    def string(self, precision=-1, ignoreOne=False, delta=0.01, namePattern='x', patternIndex=0, sympy=True):
        if self.coefficient == 0:
            return ""
        if ignoreOne and abs(self.coefficient-1) < delta and (not sympy):
            ret = ""
        else:
            if precision == -1:
                ret = str(self.coefficient)
            else:
                ret = str(round(self.coefficient, precision))
        txt = self.term
        if sympy:
            txt = txt.replace("][", "]*[")
        if self.leftExp != None:
            txt = txt.replace('[x]', "(" + self.leftExp.string(
                precision, ignoreOne, delta, namePattern, patternIndex-2) + ")")
        if namePattern != 'x1':
            txt = txt.replace('[x]', chr(ord(namePattern)+patternIndex))
            txt = txt.replace('[y]', chr(ord(namePattern)+patternIndex+1))
        else:
            txt = txt.replace('[x]', 'x{' + str(patternIndex) + "}")
            txt = txt.replace('[y]', 'x{' + str(patternIndex+1) + "}")
        if sympy:
            if txt == "":
                return ret
            else:
                return ret + "*" + txt.replace("^", "**")
        else:
            return ret + txt
