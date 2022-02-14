from bacon.nets.lsp3 import lsp3
from bacon.nets.poly2 import poly2
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import copy


class baconStack():
    def __createBacon(self, type, optimizer='adam', initializer='identity'):
        if type == "poly2":
            return poly2(optimizer, initializer)
        elif type == "lsp3":
            return lsp3(optimizer, initializer)
        else:
            raise Exception("unsupported bacon type: " + type)

    def __init__(self, size=0, bacons=None, baconType=None, baconNames=None, optimizer='adam', initializer='identity'):
        self.__bacons = []
        if baconType != None and size > 0:
            for i in range(size):
                self.__bacons.append(self.__createBacon(
                    baconType, optimizer, initializer))
        if baconNames != None:
            if len(self.__bacons) > 0:
                raise Exception("can't use both baconType and baconNames")
            for b in range(baconNames):
                self.__bacons.append(
                    self.__createBacon(b, optimizer, initializer))
        if bacons != None:
            if len(self.__bacons) > 0:
                raise Exception(
                    "bacons parameter can't be used with baconType or baconNames")
            self.__bacons = bacons.copy()
        inputs = []
        i = 3
        for b in self.__bacons:
            if len(inputs) == 0:
                inputs.append(keras.Input(
                    shape=(2,), name="input 1 and 2"))
                x = b.get_model()(inputs[len(inputs)-1])
            else:
                inputs.append(keras.Input(shape=(1,), name="input " + str(i)))
                x = layers.concatenate([x, inputs[len(inputs)-1]])
                x = b.get_model()(x)
                i += 1
        self.__model = keras.Model(inputs=inputs, outputs=[
                                   x], name="bacon-stack")
        self.__model.compile(
            optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    def fit(self, x, y, patience=20):
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
        if patience > 0:
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor='mae', patience=patience)]
        else:
            callbacks = []
        history = self.__model.fit(
            x=[inputs],
            y=[y],
            epochs=600, batch_size=10, verbose=0, callbacks=callbacks
        )
        return history

    def fit2(self, x, y, maeTarget=0.01, maxTries=-1, patience=20, verbose=False, weightShift='random'):
        counter = 1
        while True:
            history = self.fit(copy.deepcopy(x), copy.deepcopy(y), patience)
            mae = history.history['mae'][len(history.history['mae'])-1]
            if verbose:
                print(f"attempt {counter}, mae = {mae}")
            if abs(mae) <= maeTarget or maxTries > 0 and counter >= maxTries:
                return history, mae, counter
            counter += 1
            # we went down the wrong path, shuffle the weights and try again
            # TODO: would rotating weights work better?
            if weightShift == 'random' or weightShift == 'roll':
                model = self.get_model()
                weights = model.get_weights()
                if weightShift == 'random':
                    weights = [np.random.permutation(w) for w in weights]
                elif weightShift == 'roll':
                    weights = [np.roll(w, 1) for w in weights]
                model.set_weights(weights)

    def explain(self, delta=0.01):
        baconIdx = 0
        expressions = []
        for idx in range(len(self.__model.layers)):
            layer = self.__model.get_layer(index=idx)
            if type(layer) is keras.Sequential:
                expressions.append(
                    self.__bacons[baconIdx].explain(delta=delta))
                baconIdx += 1
        for i in range(len(expressions)-1):
            for j in range(len(expressions[i+1].terms)):
                expressions[i+1].terms[j].leftExp = expressions[i]
            pass
        return expressions[len(expressions)-1]

    def get_model(self):
        return self.__model
