from bacon import baconNet, dataCreator
from nets.lsp3 import lsp3
import math
import numpy as np

net = lsp3()

print("\nFeeding data for min(A,B) ... \n")
x, y = dataCreator.create(
    1000, 1, lambda x: np.minimum(x[0], x[1]))
net.fit(x[0], x[1], y)
m = net.explain()
print("Network explains: " + m.string(namePattern='A', ignoreOne=True))

print("\nFeeding data for max(A,B) ... \n")
x, y = dataCreator.create(
    1000, 1, lambda x: np.maximum(x[0], x[1]))
net.fit(x[0], x[1], y)
m = net.explain()
print("Network explains: " + m.string(namePattern='A', ignoreOne=True))

print("\nFeeding data for A*B... \n")
x, y = dataCreator.create(
    1000, 1, lambda x: np.product(np.array([x[0], x[1]]), axis=0))
net.fit(x[0], x[1], y)
m = net.explain()
print("Network explains: " + m.string(namePattern='A', ignoreOne=True))

print("\nFeeding data for (A+B)/2... \n")
x, y = dataCreator.create(
    1000, 1, lambda x: (x[0]+x[1])/2)
net.fit(x[0], x[1], y)
m = net.explain()
print("Network explains: " + m.string(namePattern='A', ignoreOne=True))

print("\n -= WAN =-\n")
