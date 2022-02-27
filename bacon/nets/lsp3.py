from bacon.net import baconNet, expression, term
import tensorflow as tf

import sys
sys.path.append("..")


class lsp3(baconNet):
    def __init__(self, optimizer='adam', initializer='identity'):
        super().__init__(3, constantTerm=False, optimizer=optimizer,
                         initializer=initializer)

    def explain_contribution(self, m, c, singleVariable=False, delta=0.01):
        if singleVariable:
            return expression(terms=[term(term="[x]", coefficient=1.0)])
        terms = []
        # minV = m[0][0]
        # maxV = m[1][0]
        # prod = m[2][0]
        # sum = abs(minV)+abs(maxV)
        # if sum > 0:
        #     minV = abs(minV) / sum
        #     maxV = abs(maxV) / sum
        # if abs(minV-1) < delta:
        #     return expression(terms=[term(term="min([x],[y])", coefficient=1.0)])
        # if abs(maxV-1) < delta:
        #     return expression(terms=[term(term="max([x],[y])", coefficient=1.0)])
        # if abs(minV-0.5) < delta and abs(maxV-0.5) < delta:
        #     return expression(terms=[term(term="([x]+[y])/2", coefficient=1.0)])
        # if abs(prod-1) < delta:
        #     return expression(terms=[term(term="[x]*[y]", coefficient=1.0)])
        # print(m)
        # return expression(terms=[term(term="can't explain", coefficient=1.0)])
        xV = m[0][0]
        yV = m[1][0]
        xyV = m[2][0]
        if abs(xV-1) < delta and abs(yV-1) < delta and abs(xyV-1) < delta:
            return expression(terms=[term(term="max([x],[y])", coefficient=1.0)])
        elif abs(xV-1) < delta and abs(yV-1) < delta and abs(xyV+1) < delta:
            return expression(terms=[term(term="min([x],[y])", coefficient=1.0)])
        elif abs(abs(xV+xyV)-1) < delta or abs(yV-xyV) < delta or abs(xV-xyV) < delta or abs(abs(yV+xyV)-1) < delta:
            return expression(terms=[term(term="max([x],[y])", coefficient=1.0)])
        elif abs(xV+xyV) < delta or abs(abs(yV-xyV)-1) < delta or abs(abs(xV-xyV)-1) < delta or abs(yV+xyV) < delta:
            return expression(terms=[term(term="min([x],[y])", coefficient=1.0)])
        print(m)
        return expression(terms=[term(term="can't explain", coefficient=1.0)])

    def expand(self, a, b):
        X = tf.stack((tf.cast(0.5*a, dtype='float32'),
                      tf.cast(0.5*b, dtype='float32'),
                      tf.cast(tf.math.abs(0.5*a-0.5*b), dtype='float32')))
        return tf.transpose(X)
