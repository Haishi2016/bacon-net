from bacon import dataCreator
from stack import baconStack
import math
import numpy as np

stack = baconStack(size=3, baconType="poly2")

x, y = dataCreator.create(
    5000, 1, lambda x: x[0] + x[1] + x[2]+x[3], params=4)
history = stack.fit(x, y)
