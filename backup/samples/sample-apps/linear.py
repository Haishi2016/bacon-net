from bacon.net import dataCreator
from bacon.stack import baconStack
import math
import numpy as np

stack1 = baconStack(size=1, baconType="linear")

# x, y = dataCreator.create(1000, 1, lambda x:
#                           np.sqrt(x[0] * x[0] + x[1] * x[1]), params=2)

params = 2
size = 100000
scale = 1
x = []
x.append(np.random.rand(size) * 0.9+0.1)
x.append(x[0]-0.1)
# x.append(np.random.rand(size))
# x.append(np.random.rand(size))
y = np.sqrt(x[0]*x[0]+x[1]*x[1])


info = stack1.fit2(x, y, maeTarget=0.003, verbose=True, weightShift='random')
m1 = stack1.explain()
print(m1.string(precision=4))
print(m1.simplify(m1.string(precision=4)))
