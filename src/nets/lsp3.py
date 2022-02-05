from bacon import baconNet
import numpy as np

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

    def expand(self, a, b, y):
        X = np.column_stack((np.minimum(a, b),
                             np.maximum(a, b),
                             np.product(np.array([a, b]), axis=0)))
        return np.column_stack((a, b)), X, y
