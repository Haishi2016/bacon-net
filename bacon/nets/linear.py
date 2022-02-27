from lib2to3.pgen2.token import LEFTSHIFT
from bacon.net import baconNet, expression, term
import tensorflow as tf

import sys
sys.path.append("..")


class linear(baconNet):
    def __init__(self, optimizer='adam', initializer='identity'):
        super().__init__(2, optimizer=optimizer, initializer=initializer)

    def explain_contribution(self, m, c, singleVariable=False, delta=0.01):
        a = 1
        b = 1
        if singleVariable:
            b = 0
        terms = []
        if abs(m[0][0]) > delta and abs(a) > delta:
            terms.append(term(coefficient=m[0][0], term="[x]"))
        if abs(m[1][0]) > delta and abs(b) > delta:
            terms.append(term(coefficient=m[1][0], term="[y]"))
        return expression(terms)

    def expand(self, a, b):
        X = tf.stack((tf.cast(a, dtype='float32'),
                      tf.cast(b, dtype='float32')))
        return tf.transpose(X)
