import sympy

from bacon.tools.expression import simplify_expression


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