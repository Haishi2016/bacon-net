from bacon.net import dataCreator
from bacon.stack import baconStack
import numpy as np
import tensorflow as tf

print("\nUsing a 3-variable poly2 stack ... \n")

stack1 = baconStack(size=2, baconType="poly2")

x1, y1 = dataCreator.create(
    500, 1, lambda x: (x[0] + 3 * x[2]) * 2 * x[1], params=3)
info = stack1.fit2(x1, y1, maeTarget=0.002, verbose=True, weightShift='random')
print(
    f"trained after {info['attempts']} tries, mae={info['mae']}, duration={info['duration']}")
m1 = stack1.explain()
print(m1.string(precision=2))
print(m1.simplify(m1.string(precision=2)))

paramCount = 7

print(f"\nUsing a {paramCount}-variable lsp3 stack ... \n")

stack2 = baconStack(size=paramCount-1, baconType="lsp3",
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=0.001),
                    initializer='identity')

x2, y2 = dataCreator.create(
    100, 1, lambda x: np.maximum(np.maximum(np.minimum(np.maximum(np.minimum(np.maximum(x[0], x[1]), x[2]), x[3]), x[4]), x[5]), x[6]), params=paramCount)
info = stack2.fit2(
    x2, y2, maeTarget=0.005, verbose=True, weightShift='random', inputShift='off')
print(
    f"trained after {info['attempts']} tries, mae={info['mae']}, duration={info['duration']}")
m2 = stack2.explain(delta=0.5)
print(m2.string(ignoreOne=True, sympy=False))
