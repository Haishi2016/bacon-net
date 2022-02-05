from bacon import baconNet, dataCreator
from nets.poly2 import poly2
import math
import numpy as np

net = poly2()

# rediscover formula to calculate area of circle:  a = pi * r^2
print("\nRediscover formula to calculate area of circle ... \n")
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: math.pi * a * a, singleVariable=True)
net.fit(a, b, y)
m = net.explain(singleVariable=True)
print(m)
a = net.predict(1)
print("area of circle with radius 1 = " + str(a))
b = net.predict([1, 2, 3])
print("area of circles with radius 1,2,3 = ", b)

# rediscover formula to calculate area of ellipse:  a = pi * a * b
print("\nRediscover formula to calculate area of ellipse ... \n")
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: math.pi * a * b)
net.fit(a, b, y)
m = net.explain()
print(m)

# rediscover freefall formula: d = 1/2 * G * t^2
print("\nRediscover Newton's equation of motion (initial speed = 5, acceleration = 9.81)... \n")
G = 9.81
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: 0.5 * G * a * a + 5*a, singleVariable=True)
net.fit(a, b, y)
m = net.explain(singleVariable=True)
print(m)

# rediscover (x+2y)^2+5
print("\nRediscover (x+2y)^2+5 ... \n")
a, b, y = dataCreator.create(
    1000, 1, lambda a, b: np.power(a+2*b, 2) + 5)
net.fit(a, b, y)
m = net.explain()
print(m)

print("\n -= WAN =-\n")
