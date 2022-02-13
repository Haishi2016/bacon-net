from bacon.net import dataCreator
from bacon.stack import baconStack
import numpy as np
import tensorflow as tf

print("\nUsing a 3-variable poly2 stack ... \n")

stack1 = baconStack(size=2, baconType="poly2")

x1, y1 = dataCreator.create(
    200, 1, lambda x: x[0] + x[1] + 2 * x[2], params=3)
stack1.fit2(x1, y1, maeTarget=0.001)
m1 = stack1.explain()
print(m1.string(precision=2))
print(m1.simplify(m1.string(precision=2)))

paramCount = 5

print(f"\nUsing a {paramCount}-variable lsp3 stack ... \n")

stack2 = baconStack(size=paramCount-1, baconType="lsp3",
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=0.001),
                    initializer='identity')

x2, y2 = dataCreator.create(
    5000, 1, lambda x: np.maximum(np.minimum(np.maximum(np.minimum(x[0], x[1]), x[2]), x[3]), x[4]), params=paramCount)
history, mae, counter = stack2.fit2(x2, y2, maeTarget=0.001, patience=10)
print(f"trained after {counter} tries, mae={mae}")
m2 = stack2.explain(delta=0.02)
print(m2.string(ignoreOne=True, sympy=False))
