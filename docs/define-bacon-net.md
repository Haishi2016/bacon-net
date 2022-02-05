# Define a new Bacon-Net

To define a new Bacon-Net, you need to create a new Python class that inherits the `bacon.baconNet` class, which defines a number of abstract methods that you need to implement.

## Implement the constructor

Your class constructor is supposed to invoke the superclass constructor with a parameter indicating how many candidate terms in your namespace (see the `expand()` method below). For example:

```python
def __init__(self):
  super().__init__(6)
```

## Implement the `expand()` method

The `expand()` method defines the term space you want to search. For example, for 2-degree 2-variable polynomial expressions, you would need to expand the two given input `a` and `b` to a matrix with columns: `a`, `b`, `a^2`, `b^2`, `a*b`, and a constant column, which is commonly filled with `numpy.ones()`.

The method returns the original input matrix, the expanded matrix, as well as the original output. For example:

```python
def expand(self, a, b, y):
  X = np.column_stack((a, b,
    np.power(a, 2),
    np.power(b, 2),
    np.product(np.array([a, b]), axis=0),
    np.ones(len(a))))
  return np.column_stack((a, b)), X, y
```

## Implement the `get_contribution()` method

In most cases, your `get_contribution()` method simply calls the superclass method:

```python
def get_contribution(self, a, b):
    return super().get_contribution(a, b)
```

This superclass method implementation returns the weights of _n_ terms in a `(n, 1)` matrix, plus the constant term as a number. For example, for search space `[a, b, a^2, b^2, ab, c]` and expression `(a+2b)^2+5`, the method returns `[[0],[0],[1],[4],[4],[k]],5`. Note the `k` value returned in the first matrix should be ignored, and the second returned value should be used as the constant term in the final expression.

## Implement the `explain()` method

The `explain()` method uses the result from the `get_contribution()` method and generates a string explanation of the network. It takes an optional `singleVariable` parameter that indicates if the expression should be a single-variable expression.

The logic in the `explain()` method should be a straightforward mapping of the `get_contribution()` results.
