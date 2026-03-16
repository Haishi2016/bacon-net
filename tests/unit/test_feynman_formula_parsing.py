import numpy as np
from types import SimpleNamespace

from samples.feynman.main import (
    ProblemSpec,
    build_numpy_evaluator,
    compute_operator_complexity,
    compute_selection_score,
    formula_requires_div_operator,
    parse_formula_expression,
    resolve_search_settings,
    should_use_operator_curriculum,
)


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


def test_resolve_search_settings_enables_pysr_defaults():
    settings = resolve_search_settings(
        SimpleNamespace(
            pysr_mode=True,
            operator_curriculum=False,
            operator_curriculum_epochs=0,
            complexity_weight=0.0,
            epochs=1200,
        )
    )

    assert settings.use_operator_curriculum is True
    assert settings.operator_curriculum_epochs == 240
    assert settings.complexity_weight == 0.001
    assert settings.use_candidate_ranking is True
    assert settings.selection_r2_tolerance == 0.0025


def test_compute_operator_complexity_penalizes_division_more_than_multiply():
    assert compute_operator_complexity(["add", "mul", "identity"]) < compute_operator_complexity(["add", "div", "identity"])


def test_compute_selection_score_prefers_simpler_model_when_validation_is_close():
    simple_score = compute_selection_score(validation_r2=0.97, model_complexity=3.0, complexity_weight=0.01)
    complex_score = compute_selection_score(validation_r2=0.975, model_complexity=8.0, complexity_weight=0.01)

    assert simple_score > complex_score


def test_should_use_operator_curriculum_only_for_harder_subtractive_cases():
    settings = resolve_search_settings(
        SimpleNamespace(
            pysr_mode=True,
            operator_curriculum=False,
            operator_curriculum_epochs=0,
            complexity_weight=0.0,
            epochs=1200,
        )
    )

    assert should_use_operator_curriculum(3, ["sub", "mul", "div", "identity"], settings) is True
    assert should_use_operator_curriculum(3, ["add", "mul", "div", "identity"], settings) is False
    assert should_use_operator_curriculum(2, ["sub", "mul", "div", "identity"], settings) is False