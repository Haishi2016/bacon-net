from bacon import baconNet, dataCreator
from nets.lsp3 import lsp3
import math
import numpy as np

net = lsp3()

print("\nFeeding data for min(A,B) ... \n")
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: np.minimum(a, b))
net.fit(a, b, y)
m = net.explain()
print("Network explains: " + m)

print("\nFeeding data for max(A,B) ... \n")
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: np.maximum(a, b))
net.fit(a, b, y)
m = net.explain()
print("Network explains: " + m)

print("\nFeeding data for A*B... \n")
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: np.product(np.array([a, b]), axis=0))
net.fit(a, b, y)
m = net.explain()
print("Network explains: " + m)

print("\nFeeding data for (A+B)/2... \n")
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: (a+b)/2)
net.fit(a, b, y)
m = net.explain()
print("Network explains: " + m)

print("\n -= WAN =-\n")
