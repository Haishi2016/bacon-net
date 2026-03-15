import itertools
import random

import pytest
import torch

from bacon.baconNet import baconNet


def _set_deterministic_seed(seed: int = 7) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _truth_table_from_expr(expr_fn):
    rows = list(itertools.product([0, 1], repeat=3))
    x = torch.tensor(rows, dtype=torch.float32)
    y = torch.tensor([[int(expr_fn(a, b, c))] for a, b, c in rows], dtype=torch.float32)
    return x, y


def _train_boolean_model(x, y, *, tree_layout: str, aggregator: str, seed: int = 7):
    _set_deterministic_seed(seed)

    if tree_layout == "full":
        model = baconNet(
            3,
            tree_layout="full",
            aggregator=aggregator,
            weight_mode="fixed",
            loss_amplifier=200,
            normalize_andness=False,
            loss_weight_operator_sparsity=3.0,
            loss_weight_operator_l2=0.01,
            lr_aggregator=0.03,
            lr_other=0.03,
            use_class_weighting=False,
            full_tree_max_egress=1,
            loss_weight_full_tree_egress=0.2,
            use_permutation_layer=False,
        )
        _, best_metric = model.find_best_model(
            x,
            y,
            x,
            y,
            attempts=3,
            max_epochs=1200,
            acceptance_threshold=0.95,
            save_path=None,
            operator_initial_tau=2.0,
            operator_final_tau=0.2,
            operator_freeze_epochs=0,
            full_tree_egress_warmup_epochs=150,
            full_tree_egress_ramp_epochs=300,
            full_tree_egress_start_metric=0.99,
            full_tree_egress_drop_tolerance=0.02,
            full_tree_egress_adapt_rate=0.2,
            save_model=False,
        )
        model.assembler.harden_full_tree(mode="auto")
        if hasattr(model.assembler.aggregator, "harden_operators"):
            model.assembler.aggregator.harden_operators()
    else:
        model = baconNet(
            3,
            tree_layout="left",
            aggregator=aggregator,
            weight_mode="fixed",
            loss_amplifier=1000,
            normalize_andness=False,
            use_permutation_layer=False,
        )
        _, best_metric = model.find_best_model(
            x,
            y,
            x,
            y,
            attempts=3,
            max_epochs=800,
            acceptance_threshold=0.95,
            save_path=None,
            save_model=False,
        )

    final_metric = (model.inference(x, threshold=0.5).eq(y)).float().mean().item()
    return model, best_metric, final_metric


@pytest.mark.integration
@pytest.mark.parametrize(
    ("expr_name", "expr_fn"),
    [
        ("or_and", lambda a, b, c: (a or b) and c),
        ("and_or", lambda a, b, c: (a and b) or c),
        ("or_or", lambda a, b, c: (a or b) or c),
    ],
)
def test_full_tree_logic_learns_boolean_truth_tables(expr_name, expr_fn):
    x, y = _truth_table_from_expr(expr_fn)

    model, best_metric, final_metric = _train_boolean_model(
        x,
        y,
        tree_layout="full",
        aggregator="math.operator_set.logic",
    )

    assert best_metric == pytest.approx(1.0, abs=1e-6), expr_name
    assert final_metric == pytest.approx(1.0, abs=1e-6), expr_name

    structure = model.assembler.get_full_tree_structure()
    egress_counts = {}
    for edge in structure["edges"]:
        key = (edge["layer"], edge["src"])
        egress_counts[key] = egress_counts.get(key, 0) + 1

    for layer_idx in range(len(structure["layer_widths"]) - 1):
        in_width = structure["layer_widths"][layer_idx]
        for src in range(in_width):
            assert egress_counts.get((layer_idx, src), 0) == 1, (expr_name, layer_idx, src)


@pytest.mark.integration
def test_left_tree_min_max_learns_or_and_truth_table():
    x, y = _truth_table_from_expr(lambda a, b, c: (a or b) and c)

    _, best_metric, final_metric = _train_boolean_model(
        x,
        y,
        tree_layout="left",
        aggregator="bool.min_max",
    )

    assert best_metric == pytest.approx(1.0, abs=1e-6)
    assert final_metric == pytest.approx(1.0, abs=1e-6)