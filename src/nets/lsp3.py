from bacon import baconNet, expression, term
import tensorflow as tf

import sys
sys.path.append("..")


class lsp3(baconNet):
    def __init__(self):
        super().__init__(3, constantTerm=False)

    def explain_contribution(self, m, c, singleVariable=False):
        if singleVariable:
            return expression(terms=[term(term="[x]", coefficient=1.0)])
        delta = 0.01
        terms = []
        if abs(m[0][0]-1) < delta:
            return expression(terms=[term(term="min([x],[y])", coefficient=1.0)])
        if abs(m[1][0]-1) < delta:
            return expression(terms=[term(term="max([x],[y])", coefficient=1.0)])
        if abs(m[0][0]-0.5) < delta and abs(m[1][0]-0.5) < delta:
            return expression(terms=[term(term="([x]+[y])/2", coefficient=1.0)])
        if abs(m[2][0]-1) < delta:
            return expression(terms=[term(term="[x]*[y]", coefficient=1.0)])
        return expression(terms=[term(term="can't explain", coefficient=1.0)])

    def expand(self, a, b):
        X = tf.stack((tf.cast(tf.minimum(a, b), dtype='float32'),
                      tf.cast(tf.maximum(a, b), dtype='float32'),
                      tf.cast(tf.math.multiply(a, b), dtype='float32')))
        return tf.transpose(X)
