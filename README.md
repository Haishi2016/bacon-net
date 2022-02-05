# Bacon-Net

**Bacon-Net** is a neural network architecture for building fully explainable neural network for arithmetic and gradient logic expression approximation. A Bacon-Net network can be used to discover an arithmetical or a logical expression that approximates the given dataset. And the result network is precisely explainable.

This repository contains a family of 2-variable Bacon-Net implementations. Multiple Bacon-Net can be used together to expand the search space; And Bacon-Net can be stacked into a **Bacon-Stack** that handles arbitrary number of variables.

The following table presents a list of famous formulas in different fields that are re-discovered using Bacon-Net using synthetic training data. All networks in this repository are implemented using [Keras](https://keras.io/) and Python.

## Bacon-Poly2

**Bacon-Poly2** can be used to discover 1-variable or 2-variable quadratic polynomials or linear polynomials. The following table lists some samples of Bacon-Poly2 re-discovering some of the well-known geometric formulas and physics formulas.

> **NOTE**: Coefficients and constent terms will vary a little in different runs.

| Forumla                                               | Expression                                       | Bacon-Poly2 Explanation                                        |
| ----------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| Area of a circle                                      | ![area of circle](./images/circle-area.png)      | `z = 3.1416x^2`                                                |
| Area of a ellipse                                     | ![area of ellipse](./images/ellipse-area.png)    | `z = 3.1416xy `                                                |
| Newton's equation of motion (displacement)            | ![motion-equation](./images/motion-equation.png) | `z = 4.905x^2 + 5.0x` _(initial speed = 5, acceleration=9.81)_ |
| An arbitary 2-degree 2-variable polynomial expression | ![polynomial](./images/polynomial.png)           | `z = 1.0x^2 + 4.0xy + 4.0y^2 + 5.0`                            |

## Installation

Run the following command to install:

```python
pip install bacon-net
```

## Usage

```python
from nets.poly2 import poly2

# create a network from the bacon-net network family
net = poly2()

# optionally, use dataCreator to generate 3 1-dimension arrays: input 1, input 2, output
a, b, y = dataCreator.create(1000, 1, lambda a, b: math.pi * a * a, singleVariable=True)

# train the network
net.fit(a, b, y)

# explain the network
m = net.explain(singleVariable=True)
```

# Developing Bacon-Net

To install baconnet, along with the tools you need to develop and run tests, run the following command:

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

![Bacon-Net](./images/bacon-net.png)

- **Input layer** contains two variables. For gradient logic expressions.
- **Expansion layer** defines the search space. Each node in this layer represents a candidate expression for the final approximation. Obviously, it’s desirable to have minimum overlaps among the function curves.

- **Interpolation layer** creates a linear interpolation of candidate terms from the expansion layer by adjusting weights associated with candidates.

- **Aggregation layer** calculate the interpolation result, which is compared to the training data.

It's also to feed the inputs to a family of Bacon-Net networks to search multiple expression spaces in parallel. A 1-active **Selection layer** is added on top to select the appropriate Bacon-Net in this case, as shown in the following diagram:

![Bacon-Net-Select](./images/bacon-net-selection.png)

## Bacon-Stack Architecture

A Bacon-Stack is recursively defined: a Bacon-Stack that handles _n_ variables (denoted as _B(n)_) is constructed by feeding variable _x(i)_ and the result of a _B(n-1)_ into a _B(2)_ network, which is a Bacon-Net, as shown in the following diagram:

![Bacon-Stack](./images/bacon-stack.png)

Bacon-Net doesn't assume variables to be commutative. To explore permutation of variable orders, a **Permutation layer** is added at the bottom of the Bacon-Stack, as shown in the following diagram:

![Bacon-Stack-selection](./images/bacon-stack-selection.png)

## Why the name "BACON"?

When I was in high school in encountered with a BASIC algorithm that used a brute-force method to discover a arithmetical expression to approximate a given dataset. I remembered the program was called “BACON”. However, it’s been unfruitful to find such references in Internet, so my memory may have failed me. Regardless, I’ve been wanting to recreate “BACON” all these years, and I finally get around to do it just now.

As I research into explainable AI, I see an opportunity to combine “BACON” with AI so that we can build some precisely explainable AI networks, plus the benefit of implementing a parallelable BACON using modern technologies.

## Upcoming Bacon-Net networks

- **Bacon-Poly3**

  For degree 3 polynomial expressions

- **Bacon-Trig2**

  For degree 2 trigonometic functions

- **Bacon-LSP6**

  Explainable gradient logic network for decision making

- **Bacon-CNN**

  Explainability layer on top of a CNN network

## Contact author

- Twitter: [@HaishiBai2010](https://twitter.com/HaishiBai2010)
- LinkedIn: [Haishi Bai](https://www.linkedin.com/in/haishi/)
