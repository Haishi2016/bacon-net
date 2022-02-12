from lib2to3.pgen2.token import LEFTSHIFT
from bacon import baconNet, expression, term
import tensorflow as tf

import sys
sys.path.append("..")


class poly2(baconNet):
    def __init__(self):
        super().__init__(6)

    def explain_contribution(self, m, c, singleVariable=False):
        a = 1
        b = 1
        if singleVariable:
            b = 0
        delta = 0.01
        terms = []
        if abs(m[2][0]) > delta and abs(a) > delta:
            terms.append(term(coefficient=m[2][0], term="[x]^2"))
        if abs(m[4][0]) > delta and abs(b*a) > delta:
            terms.append(term(coefficient=m[4][0], term="[x][y]"))
        if abs(m[3][0]) > delta and abs(b) > delta:
            terms.append(
                term(coefficient=m[3][0], term="[y]^2"))
        if abs(m[0][0]) > delta and abs(a) > delta:
            terms.append(term(coefficient=m[0][0], term="[x]"))
        if abs(m[1][0]) > delta and abs(b) > delta:
            terms.append(term(coefficient=m[1][0], term="[y]"))
        if abs(m[5][0]) > delta and abs(c) > delta:
            terms.append(term(coefficient=c))
        return expression(terms)

    def expand(self, a, b):
        X = tf.stack((tf.cast(a, dtype='float32'),
                      tf.cast(b, dtype='float32'),
                      tf.cast(tf.math.pow(a, 2), dtype='float32'),
                      tf.cast(tf.math.pow(b, 2), dtype='float32'),
                      tf.cast(tf.math.multiply(a, b), dtype='float32'),
                      tf.ones(tf.shape(a), dtype='float32')))
        return tf.transpose(X)
