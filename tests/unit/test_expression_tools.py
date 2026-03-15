import sympy
import torch

from bacon.aggregators.math import ArithmeticOperatorSet
from bacon.tools.expression import simplify_expression
from bacon.binaryTreeLogicNet import binaryTreeLogicNet


def test_simplify_expression_collapses_linear_scale_and_offset():
    expression = "(2.1486*(4.6530*c + 1.4607*b + -1.3941*a) + 0)"

    simplified = simplify_expression(expression, variables=["a", "b", "c"])

    a, b, c = sympy.symbols("a b c")
    assert sympy.expand(sympy.sympify(simplified)) == sympy.expand(sympy.sympify(expression))
    assert "+ 0" not in simplified
    assert simplified != expression


def test_simplify_expression_preserves_non_sympy_syntax():
    expression = "(a ∨ b)"

    simplified = simplify_expression(expression, variables=["a", "b"])

    assert simplified == expression


def test_binary_tree_logic_net_input_labels_include_constant_leaf():
    model = binaryTreeLogicNet(
        input_size=2,
        tree_layout="alternating",
        aggregator=ArithmeticOperatorSet(),
        use_permutation_layer=False,
        use_constant_input=True,
        device=torch.device("cpu"),
    )

    assert model.get_input_labels(["a", "b"]) == ["a", "b", "1"]
