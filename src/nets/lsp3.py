from bacon import baconNet
import tensorflow as tf

import sys
sys.path.append("..")


class lsp3(baconNet):
    def __init__(self):
        super().__init__(3, constantTerm=False)

    def get_contribution(self, a, b):
        return super().get_contribution(a, b)

    def explain(self, singleVariable=False):
        if singleVariable:
            return "A"
        m, c = self.get_contribution(1, 1)
        delta = 0.01
        terms = []
        if abs(m[0][0]-1) < delta:
            return "min(A, B)"
        if abs(m[1][0]-1) < delta:
            return "max(A, B)"
        if abs(m[0][0]-0.5) < delta and abs(m[1][0]-0.5) < delta:
            return "(A + B) / 2"
        if abs(m[2][0]-1) < delta:
            return "A * B"
        return "can't explain"

    def expand(self, a, b):
        X = tf.stack((tf.cast(tf.minimum(a, b), dtype='float32'),
                      tf.cast(tf.maximum(a, b), dtype='float32'),
                      tf.cast(tf.math.multiply(a, b), dtype='float32')))
        return tf.transpose(X)
