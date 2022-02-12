from bacon import dataCreator
from stack import baconStack

stack = baconStack(size=2, baconType="poly2")

x, y = dataCreator.create(
    5000, 1, lambda x: x[0] + x[1] + 2 * x[2], params=3)
stack.fit(x, y)
m = stack.explain()
print(m.simplify(m.string(precision=2)))
