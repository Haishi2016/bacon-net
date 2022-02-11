from nets.lsp3 import lsp3
from nets.poly2 import poly2
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf


class baconStack():
    def __createBacon(self, type):
        if type == "poly2":
            return poly2()
        elif type == "lsp3":
            return lsp3()
        else:
            raise Exception("unsupported bacon type: " + type)

    def __init__(self, size=0, bacons=None, baconType=None, baconNames=None):
        myBacons = []
        if baconType != None and size > 0:
            for i in range(size):
                myBacons.append(self.__createBacon(baconType))
        if baconNames != None:
            if len(myBacons) > 0:
                raise Exception("can't use both baconType and baconNames")
            for b in range(baconNames):
                myBacons.append(self.__createBacon(b))
        if bacons != None:
            if len(myBacons) > 0:
                raise Exception(
                    "bacons parameter can't be used with baconType or baconNames")
            myBacons = bacons.copy()
        inputs = []
        for b in myBacons:
            if len(inputs) == 0:
                inputs.append(keras.Input(shape=(2,)))
                x = b.get_model()(inputs[len(inputs)-1])
            else:
                inputs.append(keras.Input(shape=(1,)))
                x = layers.concatenate([x, inputs[len(inputs)-1]])
                x = b.get_model()(x)
        self.__model = keras.Model(inputs=inputs, outputs=[
                                   x], name="bacon-stack")
        self.__model.compile(
            optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    def fit(self, x, y):
        cols = len(x)
        rows = len(x[0])
        inputs = []
        if cols < 2:
            inputs.append(np.column_stack(
                (x[0].reshape(-1, 1).flatten(), np.zeros(rows))))
        else:
            inputs.append(np.column_stack(
                (x[0].reshape(-1, 1).flatten(), x[1].reshape(-1, 1).flatten())))
        for i in range(2, cols):
            inputs.append(x[i].reshape(-1, 1).flatten())
        callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=20)
        history = self.__model.fit(
            x=[inputs],
            y=[y],
            epochs=600, batch_size=10, verbose=0, callbacks=[callback]
        )
        return history
