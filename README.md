# Bacon-Net

**Bacon-Net** is a neural network architecture for building fully explainable neural network for arithmetic and gradient logic expression approximation. A Bacon-Net network can be used to discover an arithmetical or a logical expression that approximates the given dataset. And the result network is precisely explainable.

This repository contains a family of 2-variable Bacon-Net implementations. Multiple Bacon-Net can be used together to expand the search space; And Bacon-Net can be stacked into a **Bacon-Stack** that handles arbitrary number of variables.

The following table presents a list of famous formulas in different fields that are re-discovered using Bacon-Net using synthetic training data. All networks in this repository are implemented using [Keras](https://keras.io/) and Python.

## Bacon-Poly2

**Bacon-Poly2** can be used to discover 1-variable or 2-variable quadratic polynomials or linear polynomials. The following table lists some samples of Bacon-Poly2 re-discovering some of the well-known geometric formulas and physics formulas.

> **NOTE**: Coefficients and constent terms will vary a little in different runs.

| Formula                                               | Expression                                                                       | Bacon-Poly2 Explanation                                        |
| ----------------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Area of a circle                                      | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/circle-area.png)     | `z = 3.1416x^2`                                                |
| Area of a ellipse                                     | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/ellipse-area.png)    | `z = 3.1416xy `                                                |
| Newton's equation of motion (displacement)            | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/motion-equation.png) | `z = 4.905x^2 + 5.0x` _(initial speed = 5, acceleration=9.81)_ |
| An arbitary 2-degree 2-variable polynomial expression | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/polynomial.png)      | `z = 1.0x^2 + 4.0xy + 4.0y^2 + 5.0`                            |
| Einstein's mass-energy equivalence                    | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/e-mc2.png)           | `z = 0.0899x ` _(with c normalized to 0.299792458 m/s)_        |

## Bacon-LSP3

**Bacon-LSP3** evaluates four possible gradient logic relationships between two variables in space _I = [0, 1]_: full conjunction, full disjunction, product t-norm (medium hyperconjunction) and neutrality.

Bacon-LSP3 can be used to reason the logic behind some simple decisions, like “a face image needs to show 2 eye features AND a mouth feature”

| Relationship     | Plot                                                                       | Bacon-LSP3 Explanation |
| ---------------- | -------------------------------------------------------------------------- | ---------------------- |
| Full conjunction | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/lsp3-0.png)    | `min(A, B)`            |
| Product t-norm   | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/lsp3-1_25.png) | `A * B `               |
| Neutrality       | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/lsp3-0_5.png)  | `(A + B) / 2`          |
| Full disjunction | ![](https://github.com/Haishi2016/bacon-net/raw/main/images/lsp3-1.png)    | `max(A, B)`            |

## Installation

Run the following command to install:

```python
pip install bacon-net
```

## Sample Usage

```python
from nets.poly2 import poly2

# create a network from the bacon-net network family
net = poly2()

# optionally, use dataCreator to generate 3 1-dimension arrays: input a, input b, output (y)
# set singleVariable flag if only a single variable (a) is used
a, b, y = dataCreator.create(1000, 1, lambda a, b: math.pi * a * a, singleVariable=True)

# train the network
net.fit(a, b, y)

# explain the network
m = net.explain(singleVariable=True)

# make prediction (pass two parameters if two variables are used)
p = net.predict(2.4)

# make predictions on array (pass two arrays if two variables are used)
p = net.predict([1.0, 2.3, 4.3])
```

# Developing Bacon-Net

To install Bacon-Net, along with the tools you need to develop and run tests, run the following command:

```python
pip install -e .[dev]
```

To run all test cases, run the following command from the project's root folder:

```python
pytest
```

Please see [here](./docs/define-bacon-net.md) for instructions on creating a new Bacon-Net network.

## Bacon-Net Architecture

The idea behind Bacon-Net is simple: to construct a network that can do linear interpolation among a group of selected terms like _min(x,y)_ and _sin(x^2)_, as shown in the following diagram:

![](https://github.com/Haishi2016/bacon-net/raw/main/images/bacon-net.png)

- **Input layer** contains two variables. For gradient logic expressions.
- **Expansion layer** defines the search space. Each node in this layer represents a candidate expression for the final approximation. Obviously, it’s desirable to have minimum overlaps among the function curves.

- **Interpolation layer** creates a linear interpolation of candidate terms from the expansion layer by adjusting weights associated with candidates.

- **Aggregation layer** calculate the interpolation result, which is compared to the training data.

It's also to feed the inputs to a family of Bacon-Net networks to search multiple expression spaces in parallel. A 1-active **Selection layer** is added on top to select the appropriate Bacon-Net in this case, as shown in the following diagram:

![](https://github.com/Haishi2016/bacon-net/raw/main/images/bacon-net-selection.png)

## Bacon-Stack Architecture

A Bacon-Stack is recursively defined: a Bacon-Stack that handles _n_ variables (denoted as _B(n)_) is constructed by feeding variable _x(i)_ and the result of a _B(n-1)_ into a _B(2)_ network, which is a Bacon-Net, as shown in the following diagram:

![](https://github.com/Haishi2016/bacon-net/raw/main/images/bacon-stack.png)

Bacon-Net doesn't assume variables to be commutative. To explore permutation of variable orders, a **Permutation layer** is added at the bottom of the Bacon-Stack, as shown in the following diagram:

![](https://github.com/Haishi2016/bacon-net/raw/main/images/bacon-stack-selection.png)

## Why the name "BACON"?

When I was in high school in encountered with a BASIC program that used a brute-force method to discover a arithmetical expression to approximate a given dataset. I remembered the program was called “BACON”. However, it’s been unfruitful to find such references in Internet, so my memory may have failed me. Regardless, I’ve been wanting to recreate “BACON” all these years, and I finally got around to do it just now during my week off.

As I research into explainable AI, I see an opportunity to combine “BACON” with AI so that we can build some precisely explainable AI networks, plus the benefit of implementing a parallelable, GPU-accelerated BACON using modern technologies.

## Upcoming Bacon-Net networks

- **Bacon-Poly3**

  For degree 3 polynomial expressions

- **Bacon-Trig2**

  For degree 2 trigonometic functions

- **Bacon-LSP6**

  Explainable gradient logic network for decision making

- **Bacon-CNN**

  Explainability layer on top of a CNN network

- **Bacon-H1**

  A combination of selected Bacon-Net networks

- **Bacon-Cal1**

  A simple calculus solver

## Contact author

- Twitter: [@HaishiBai2010](https://twitter.com/HaishiBai2010)
- LinkedIn: [Haishi Bai](https://www.linkedin.com/in/haishi/)
