from bacon import baconNet, dataCreator
from nets.poly2 import poly2
import math
import numpy as np

net = poly2()

a, b, y = dataCreator.create(
    1000, 1, lambda a, b: np.power(a+b, 2) + 5)
net.fit(a, b, y)
m = net.explain(1, 1)
print(m)

# rediscover freefall formula: d = 1/2 * G * t^2
print("Rediscover freefall formula: d = 1/2 * G * t^2 ... ")
G = 9.81
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: 0.5 * G * a * a, singleVariable=True)
net.fit(a, b, y)
m = net.explain(0.5, 0)
print(m)

# rediscover formula to calculate area of circle:  a = pi * r^2
print("Rediscover formula to calculate area of circle:  a = pi * r^2 ... ")
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: math.pi * a * a, singleVariable=True)
net.fit(a, b, y)
m = net.explain(0.5, 0)
print(m)
