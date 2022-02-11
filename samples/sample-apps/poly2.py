from bacon import baconNet, dataCreator
from nets.poly2 import poly2
import math
import numpy as np

net = poly2()

# rediscover formula to calculate area of circle:  a = pi * r^2
print("\nRediscover formula to calculate area of circle ... \n")
x, y = dataCreator.create(
    1000, 1, lambda x: math.pi * x[0] * x[0], params=1)
net.fit(x[0], x[1], y)
m = net.explain(singleVariable=True)
print(m.string(4))
a = net.predict(1)
print("area of circle with radius 1 = " + str(a))
b = net.predict([1, 2, 3])
print("area of circles with radius 1,2,3 = ", b)

# rediscover formula to calculate area of ellipse:  a = pi * a * b
print("\nRediscover formula to calculate area of ellipse ... \n")
x, y = dataCreator.create(
    1000, 1, lambda x: math.pi * x[0] * x[1], params=2)
net.fit(x[0], x[1], y)
m = net.explain()
print(m.string(4))

# rediscover freefall formula: d = 1/2 * G * t^2
print("\nRediscover Newton's equation of motion (initial speed = 5, acceleration = 9.81)... \n")
G = 9.81
x, y = dataCreator.create(
    1000, 1, lambda x: 0.5 * G * x[0] * x[0] + 5 * x[0], params=1)
net.fit(x[0], x[1], y)
m = net.explain(singleVariable=True)
print(m.string(4))

# rediscover (x+2y)^2+5
print("\nRediscover (x+2y)^2+5 ... \n")
x, y = dataCreator.create(
    1000, 1, lambda x: np.power(x[0]+2*x[1], 2) + 5)
net.fit(x[0], x[1], y)
m = net.explain()
print(m.string(0))

# rediscover E = mc^2
print("\nRediscover mass-energy equivalence ... \n")
x, y = dataCreator.create(
    1000, 1, lambda x: x[0] * 0.299792458 * 0.299792458, params=1)
net.fit(x[0], x[1], y)
m = net.explain(singleVariable=True)
print(m.string(4))

print("\n -= WAN =-\n")
