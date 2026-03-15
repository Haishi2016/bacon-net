import numpy as np

from samples.feynman.main import ProblemSpec, build_numpy_evaluator, formula_requires_div_operator, parse_formula_expression


def test_parse_formula_expression_treats_gamma_as_symbol_and_pi_as_constant():
    expression = parse_formula_expression("1/(gamma-1)*pr*V")
    result = expression.subs({"gamma": 3, "pr": 5, "V": 2}).evalf()

    assert float(result) == 5.0


def test_formula_requires_div_operator_handles_reserved_names():
    assert formula_requires_div_operator("1/(gamma-1)*pr*V") is True


def test_build_numpy_evaluator_handles_gamma_and_pi_formulas():
    problem = ProblemSpec(
        problem_id="I.6.2a",
        name="I.6.2a",
        formula="exp(-theta**2/2)/sqrt(2*pi)",
        variable_names=["theta"],
        variable_ranges=[(-1.0, 1.0)],
    )

    evaluator = build_numpy_evaluator(problem)
    outputs = evaluator(np.array([0.0, 1.0]))

    assert outputs.shape == (2,)
    np.testing.assert_allclose(outputs[0], 1.0 / np.sqrt(2.0 * np.pi), rtol=1e-6, atol=1e-6)