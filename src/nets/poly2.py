from bacon import baconNet
import numpy as np

import sys
sys.path.append("..")


class poly2(baconNet):
    def __init__(self):
        super().__init__(6)

    def get_contribution(self, a, b):
        return super().get_contribution(a, b)

    def explain(self, singleVariable=False):
        a = 1
        b = 1
        if singleVariable:
            b = 0
        m, c = self.get_contribution(a, b)
        delta = 0.01
        terms = []
        if abs(m[2][0]) > delta and abs(a) > delta:
            terms.append(str(round(m[2][0], 4)) + "x^2")
        if abs(m[4][0]) > delta and abs(b*a) > delta:
            terms.append(str(round(m[4][0], 4)) + "xy")
        if abs(m[3][0]) > delta and abs(b) > delta:
            terms.append(str(round(m[3][0], 4)) + "y^2")
        if abs(m[0][0]) > delta and abs(a) > delta:
            terms.append(str(round(m[0][0], 4)) + "x")
        if abs(m[1][0]) > delta and abs(b) > delta:
            terms.append(str(round(m[1][0], 4)) + "y")
        if abs(m[5][0]) > delta and abs(c) > delta:
            terms.append(str(round(c, 4)))
        return "z = " + " + ".join(terms)

    def expand(self, a, b, y):
        X = np.column_stack((a, b,
                             np.power(a, 2),
                             np.power(b, 2),
                             np.product(np.array([a, b]), axis=0),
                             np.ones(len(a))))
        return np.column_stack((a, b)), X, y
