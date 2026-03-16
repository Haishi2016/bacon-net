#!/usr/bin/env python3
"""Run BACON against official Feynman formulas without local dataset preparation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import sympy
import torch
import torch.nn as nn
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bacon.baconNet import baconNet
from samples.common import train_bacon_model


logging.basicConfig(level=logging.INFO, format="%(message)s")

OFFICIAL_METADATA_URLS = {
    "feynman": "http://space.mit.edu/home/tegmark/aifeynman/FeynmanEquations.csv",
    "bonus": "http://space.mit.edu/home/tegmark/aifeynman/BonusEquations.csv",
}

FUNCTION_NAME_MAP = {
    "arccos": "acos",
    "arcsin": "asin",
    "arctan": "atan",
    "ln": "log",
}

SYMBOL_PATTERN = re.compile(r"\b[a-zA-Z_]\w*\b")
FUNCTION_NAMES = {
    "exp", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "log",
}
CONSTANT_SYMBOLS = {
    "pi": sympy.pi,
    "e": sympy.E,
    "E": sympy.E,
}


@dataclass
class ProblemSpec:
    problem_id: str
    name: str
    formula: str
    variable_names: list[str]
    variable_ranges: list[tuple[float, float]]


@dataclass
class BenchmarkResult:
    problem_id: str
    name: str
    validation_r2: float
    test_r2: float
    operator_set: list[str]
    selected_operators: list[str]
    alternating_axb: bool
    success: bool
    model_complexity: float = 0.0
    selection_score: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class SearchSettings:
    use_operator_curriculum: bool
    operator_curriculum_epochs: int
    complexity_weight: float
    use_candidate_ranking: bool
    selection_r2_tolerance: float


@dataclass
class CandidateResult:
    model: baconNet
    validation_r2: float
    test_r2: float
    selected_operators: list[str]
    model_complexity: float
    selection_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BACON on official Feynman formulas.")
    parser.add_argument("--dataset", choices=["bonus", "feynman"], default="feynman")
    parser.add_argument("--problem", action="append", dest="problems", help="Problem ID to run, e.g. I.13.4")
    parser.add_argument("--all", action="store_true", help="Run all problems in the selected dataset")
    parser.add_argument("--tree-layout", choices=["full", "alternating"], default="alternating")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--samples-train", type=int, default=1000)
    parser.add_argument("--samples-val", type=int, default=200)
    parser.add_argument("--samples-test", type=int, default=200)
    parser.add_argument("--operator-mode", choices=["auto", "basic", "expanded"], default="auto")
    parser.add_argument("--pysr-mode", action="store_true", help="Use PySR-inspired search heuristics: staged operator curriculum plus complexity-aware candidate ranking")
    parser.add_argument("--complexity-weight", type=float, default=0.0, help="Penalty weight used when ranking candidate models by validation R² minus complexity")
    parser.add_argument("--operator-curriculum", action="store_true", help="Freeze operators to a safe default early in training, then unlock the full operator set")
    parser.add_argument("--operator-curriculum-epochs", type=int, default=0, help="Number of warmup epochs for the staged operator curriculum; 0 picks a heuristic default when enabled")
    parser.add_argument("--alternating-axb", action="store_true", help="Enable alternating coefficient layers of the form a*x^b with b constrained to [1, 2]")
    parser.add_argument("--axb-reg-weight", type=float, default=0.0, help="Optional regularization weight that encourages b to snap toward the allowed endpoints")
    parser.add_argument("--use-constant-input", action="store_true", help="Append a learned constant leaf with value 1 after routing/transforms so the tree can form offsets like 1+x")
    parser.add_argument("--success-threshold", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the sweep immediately if one problem fails instead of recording the failure and continuing")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metadata_cache_path(dataset: str) -> Path:
    return Path(__file__).resolve().parent / ".cache" / f"{dataset}_equations.csv"


def ensure_metadata_csv(dataset: str) -> Path:
    cache_path = metadata_cache_path(dataset)
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = OFFICIAL_METADATA_URLS[dataset]
    logging.info(f"⬇️  Downloading {dataset} metadata from {url}")
    try:
        with urlopen(url, timeout=60) as response:
            cache_path.write_bytes(response.read())
    except URLError:
        curl_executable = shutil.which("curl") or shutil.which("curl.exe")
        if curl_executable is None:
            raise
        completed = subprocess.run(
            [curl_executable, "-L", url, "-o", str(cache_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not cache_path.exists():
            raise RuntimeError(f"Failed to download metadata via curl: {completed.stderr.strip()}")
    return cache_path


def normalize_formula(formula: str) -> str:
    normalized = formula.strip()
    for source_name, target_name in FUNCTION_NAME_MAP.items():
        normalized = normalized.replace(source_name, target_name)
    return normalized


def build_formula_locals(formula: str, variable_names: list[str] | None = None) -> dict[str, sympy.Basic]:
    local_symbols: dict[str, sympy.Basic] = dict(CONSTANT_SYMBOLS)
    if variable_names is not None:
        for name in variable_names:
            local_symbols[name] = sympy.Symbol(name)

    for name in SYMBOL_PATTERN.findall(formula):
        if name in local_symbols:
            continue
        if name.lower() in FUNCTION_NAMES:
            continue
        local_symbols[name] = sympy.Symbol(name)

    return local_symbols


def parse_formula_expression(formula: str, variable_names: list[str] | None = None) -> sympy.Expr:
    normalized = normalize_formula(formula)
    local_symbols = build_formula_locals(normalized, variable_names=variable_names)
    return sympy.sympify(normalized, locals=local_symbols)


def load_problem_specs(dataset: str) -> list[ProblemSpec]:
    csv_path = ensure_metadata_csv(dataset)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        problems: list[ProblemSpec] = []
        for row in reader:
            problem_id = (row.get("Filename") or row.get("\ufeffFilename") or "").strip()
            formula = (row.get("Formula") or "").strip()
            if not problem_id or not formula:
                continue

            variable_names: list[str] = []
            variable_ranges: list[tuple[float, float]] = []
            for idx in range(1, 11):
                name = (row.get(f"v{idx}_name") or "").strip()
                if not name:
                    continue
                low_raw = (row.get(f"v{idx}_low") or "").strip()
                high_raw = (row.get(f"v{idx}_high") or "").strip()
                if not low_raw or not high_raw:
                    continue
                variable_names.append(name)
                variable_ranges.append((float(low_raw), float(high_raw)))

            if variable_names:
                problems.append(
                    ProblemSpec(
                        problem_id=problem_id,
                        name=(row.get("Name") or problem_id).strip(),
                        formula=normalize_formula(formula),
                        variable_names=variable_names,
                        variable_ranges=variable_ranges,
                    )
                )
    return problems


def select_problems(problem_specs: list[ProblemSpec], requested: list[str] | None, run_all: bool, max_problems: int | None) -> list[ProblemSpec]:
    if requested:
        wanted = {item.strip() for item in requested}
        selected = [problem for problem in problem_specs if problem.problem_id in wanted]
        missing = wanted.difference({problem.problem_id for problem in selected})
        if missing:
            raise ValueError(f"Unknown problem ids: {sorted(missing)}")
    elif run_all:
        selected = list(problem_specs)
    else:
        raise ValueError("Specify --problem <id> or --all")

    if max_problems is not None:
        selected = selected[:max_problems]
    return selected


def build_numpy_evaluator(problem: ProblemSpec) -> Callable[..., np.ndarray]:
    symbols = sympy.symbols(problem.variable_names)
    expression = parse_formula_expression(problem.formula, variable_names=problem.variable_names)
    return sympy.lambdify(symbols, expression, modules=["numpy"])


def formula_requires_div_operator(formula: str) -> bool:
    expression = parse_formula_expression(formula)
    _, denominator = sympy.fraction(sympy.together(expression))
    return bool(denominator.free_symbols)


def infer_operator_names(formula: str, mode: str, alternating_axb: bool, tree_layout: str) -> list[str]:
    if mode == "basic":
        operator_names = ["add", "sub", "mul", "div", "identity"]
    elif mode == "expanded":
        operator_names = ["add", "sub", "mul", "div", "identity"]
    else:
        compact = formula.replace(" ", "")
        operator_names: list[str] = []
        if "+" in compact:
            operator_names.append("add")
        if "-" in compact[1:]:
            operator_names.append("sub")
        if "*" in compact:
            operator_names.append("mul")
        if "/" in compact and formula_requires_div_operator(formula):
            operator_names.append("div")
        operator_names.append("identity")

    deduped: list[str] = []
    for name in operator_names:
        if name not in deduped:
            deduped.append(name)

    return deduped or ["add", "identity"]


def resolve_search_settings(args: argparse.Namespace) -> SearchSettings:
    use_operator_curriculum = bool(args.operator_curriculum or args.pysr_mode)
    operator_curriculum_epochs = int(args.operator_curriculum_epochs)
    if use_operator_curriculum and operator_curriculum_epochs <= 0:
        operator_curriculum_epochs = min(400, max(100, args.epochs // 5))

    complexity_weight = float(args.complexity_weight)
    if args.pysr_mode and complexity_weight <= 0.0:
        complexity_weight = 0.001

    use_candidate_ranking = bool(args.pysr_mode or complexity_weight > 0.0)
    return SearchSettings(
        use_operator_curriculum=use_operator_curriculum,
        operator_curriculum_epochs=operator_curriculum_epochs,
        complexity_weight=complexity_weight,
        use_candidate_ranking=use_candidate_ranking,
        selection_r2_tolerance=0.0025 if args.pysr_mode else 0.0,
    )


def get_selected_operator_names(model: baconNet) -> list[str]:
    aggregator = model.assembler.aggregator
    if not hasattr(aggregator, "op_logits_per_node") or aggregator.op_logits_per_node is None:
        return []

    selected: list[str] = []
    for logits in aggregator.op_logits_per_node:
        probs = torch.softmax(logits, dim=0)
        selected.append(aggregator.op_names[int(torch.argmax(probs).item())])
    return selected


def compute_operator_complexity(selected_operators: list[str]) -> float:
    operator_weights = {
        "identity": 0.0,
        "add": 0.5,
        "sub": 0.75,
        "mul": 1.5,
        "div": 2.5,
    }
    return float(sum(operator_weights.get(name, 1.5) for name in selected_operators))


def compute_model_complexity(model: baconNet) -> float:
    complexity = compute_operator_complexity(get_selected_operator_names(model))

    if getattr(model.assembler, "use_constant_input", False):
        complexity += 1.0

    alternating_tree = getattr(model.assembler, "alternating_tree", None)
    if alternating_tree is not None:
        for coeff_layer in alternating_tree.coeff_layers:
            if getattr(coeff_layer, "learn_exponents", False):
                exponents = coeff_layer.get_exponents().detach().cpu()
                complexity += float(torch.abs(exponents - 1.0).sum().item() * 0.5)

    return complexity


def compute_selection_score(validation_r2: float, model_complexity: float, complexity_weight: float) -> float:
    return float(validation_r2 - complexity_weight * model_complexity)


def should_use_operator_curriculum(num_inputs: int, operator_names: list[str], search_settings: SearchSettings) -> bool:
    if not search_settings.use_operator_curriculum:
        return False
    if num_inputs < 3:
        return False
    return "sub" in operator_names


def sample_problem_data(
    problem: ProblemSpec,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    evaluator = build_numpy_evaluator(problem)
    total_size = train_size + val_size + test_size
    rng = np.random.default_rng(seed)

    batches_x: list[np.ndarray] = []
    batches_y: list[np.ndarray] = []
    remaining = total_size
    attempts = 0

    while remaining > 0 and attempts < 10:
        attempts += 1
        batch_size = max(remaining * 3, remaining)
        columns = [rng.uniform(low, high, size=batch_size) for low, high in problem.variable_ranges]
        values = np.asarray(evaluator(*columns), dtype=np.float64)
        if values.ndim == 0:
            values = np.full(batch_size, values, dtype=np.float64)
        x_matrix = np.stack(columns, axis=1)
        mask = np.isfinite(values)
        if mask.ndim > 1:
            mask = mask.reshape(-1)
        if mask.any():
            batches_x.append(x_matrix[mask][:remaining])
            batches_y.append(values[mask][:remaining])
            remaining = total_size - sum(len(chunk) for chunk in batches_y)

    if remaining > 0:
        raise RuntimeError(f"Could not sample enough finite points for {problem.problem_id}")

    x_tensor = torch.tensor(np.concatenate(batches_x, axis=0)[:total_size], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(np.concatenate(batches_y, axis=0)[:total_size], dtype=torch.float32, device=device).unsqueeze(1)

    train_end = train_size
    val_end = train_size + val_size
    return (
        x_tensor[:train_end],
        y_tensor[:train_end],
        x_tensor[train_end:val_end],
        y_tensor[train_end:val_end],
        x_tensor[val_end:],
        y_tensor[val_end:],
    )


def choose_full_tree_depth(num_inputs: int) -> int:
    return max(2, min(num_inputs - 1, 4))


def configure_operator_set(model: baconNet, operator_names: list[str], device: torch.device) -> None:
    aggregator = model.assembler.aggregator
    aggregator.op_names = list(operator_names)
    aggregator.num_ops = len(operator_names)
    if model.assembler.tree_layout == "alternating" and model.assembler.alternating_tree is not None:
        num_agg_nodes = model.assembler.alternating_tree.num_agg_nodes
    elif model.assembler.tree_layout == "full" and model.assembler.fully_connected_tree is not None:
        num_agg_nodes = sum(model.assembler.fully_connected_tree.layer_widths[1:])
    else:
        num_agg_nodes = model.assembler.num_layers
    aggregator.op_logits_per_node = nn.ParameterList(
        [nn.Parameter(torch.zeros(len(operator_names), device=device)) for _ in range(num_agg_nodes)]
    )
    aggregator.num_layers = num_agg_nodes
    aggregator.use_gumbel = True


def create_regression_model(
    num_inputs: int,
    operator_names: list[str],
    tree_layout: str,
    alternating_axb: bool,
    axb_reg_weight: float,
    use_constant_input: bool,
    device: torch.device,
) -> baconNet:
    model = baconNet(
        input_size=num_inputs,
        tree_layout=tree_layout,
        aggregator="math.operator_set.arith",
        weight_mode="trainable",
        weight_normalization="none",
        loss_amplifier=1.0,
        normalize_andness=False,
        use_transformation_layer=False,
        use_class_weighting=False,
        permutation_initial_temperature=5.0,
        permutation_final_temperature=0.5,
        lr_aggregator=0.01,
        lr_other=0.01,
        regression_loss_type="mse",
        loss_weight_operator_sparsity=0.1,
        loss_weight_operator_l2=0.0,
        full_tree_depth=choose_full_tree_depth(num_inputs),
        full_tree_temperature=10.0,
        full_tree_final_temperature=0.1,
        full_tree_shape="triangle",
        full_tree_max_egress=1,
        loss_weight_full_tree_egress=0.5,
        loss_weight_full_tree_ingress=0.0,
        loss_weight_full_tree_ingress_balance=50.0,
        loss_weight_full_tree_scale_reg=0.5,
        full_tree_concentrate_ingress=False,
        full_tree_use_sinkhorn=False,
        alternating_learn_first_routing=(tree_layout == "alternating"),
        alternating_learn_subsequent_routing=(tree_layout == "alternating"),
        alternating_learn_exponents=alternating_axb,
        alternating_min_exponent=1.0,
        alternating_max_exponent=2.0,
        alternating_max_egress=1,
        alternating_use_straight_through=True,
        loss_weight_alternating_balance=0.1,
        loss_weight_alternating_egress=0.5,
        loss_weight_alternating_exponent_reg=axb_reg_weight,
        use_constant_input=use_constant_input,
        use_permutation_layer=False,
    )
    configure_operator_set(model, operator_names, device)
    return model


def compute_r2(model: baconNet, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        ss_res = ((y - predictions) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return float((1 - ss_res / (ss_tot + 1e-8)).item())


def harden_trained_model(model: baconNet, tree_layout: str, variable_names: list[str]) -> list[str]:
    if tree_layout == "alternating" and model.assembler.alternating_tree is not None:
        model.assembler.harden_alternating_tree()
        log_alternating_exponents(model, variable_names)
    elif tree_layout == "full" and model.assembler.fully_connected_tree is not None:
        model.assembler.harden_full_tree(mode="auto")

    if hasattr(model.assembler.aggregator, "harden_operators"):
        model.assembler.aggregator.harden_operators()

    return get_selected_operator_names(model)


def train_candidate(
    problem: ProblemSpec,
    args: argparse.Namespace,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    operator_names: list[str],
    search_settings: SearchSettings,
    attempt_index: int,
) -> CandidateResult:
    attempt_seed = args.seed + attempt_index
    torch.manual_seed(attempt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(attempt_seed)

    model = create_regression_model(
        num_inputs=len(problem.variable_names),
        operator_names=operator_names,
        tree_layout=args.tree_layout,
        alternating_axb=args.alternating_axb,
        axb_reg_weight=args.axb_reg_weight,
        use_constant_input=args.use_constant_input,
        device=device,
    )

    use_operator_curriculum = should_use_operator_curriculum(
        num_inputs=len(problem.variable_names),
        operator_names=operator_names,
        search_settings=search_settings,
    )

    best_model, best_val_r2 = train_bacon_model(
        model=model,
        X_train=x_train,
        Y_train=y_train,
        X_test=x_val,
        Y_test=y_val,
        attempts=1,
        max_epochs=args.epochs,
        acceptance_threshold=1.0,
        task_type="regression",
        use_hierarchical_permutation=False,
        operator_initial_tau=3.0,
        operator_final_tau=0.1,
        operator_freeze_min_confidence=0.85,
        operator_freeze_epochs=search_settings.operator_curriculum_epochs if use_operator_curriculum else 0,
        frozen_training_epochs=min(800, max(200, args.epochs // 2)),
        save_model=False,
        save_path=None,
        full_tree_egress_warmup_epochs=min(150, max(50, args.epochs // 12)),
        full_tree_egress_ramp_epochs=min(300, max(100, args.epochs // 6)),
        full_tree_egress_start_metric=0.99,
        full_tree_egress_drop_tolerance=0.02,
        full_tree_egress_adapt_rate=0.2,
    )

    selected_operators = harden_trained_model(best_model, args.tree_layout, problem.variable_names)
    test_r2 = compute_r2(best_model, x_test, y_test)
    model_complexity = compute_model_complexity(best_model)
    selection_score = compute_selection_score(best_val_r2, model_complexity, search_settings.complexity_weight)

    return CandidateResult(
        model=best_model,
        validation_r2=float(best_val_r2),
        test_r2=float(test_r2),
        selected_operators=selected_operators,
        model_complexity=float(model_complexity),
        selection_score=float(selection_score),
    )


def choose_best_candidate(candidates: list[CandidateResult], search_settings: SearchSettings) -> CandidateResult:
    best_validation_r2 = max(item.validation_r2 for item in candidates)
    near_best = [
        item for item in candidates
        if item.validation_r2 >= best_validation_r2 - search_settings.selection_r2_tolerance
    ]
    return max(near_best, key=lambda item: (item.selection_score, item.validation_r2, -item.model_complexity))


def log_alternating_exponents(model: baconNet, variable_names: list[str]) -> None:
    alternating_tree = getattr(model.assembler, "alternating_tree", None)
    if alternating_tree is None:
        return
    for layer_idx, coeff_layer in enumerate(alternating_tree.coeff_layers):
        if not getattr(coeff_layer, "learn_exponents", False):
            continue
        coeffs = coeff_layer.get_coefficients().cpu().tolist()
        exponents = coeff_layer.get_exponents().cpu().tolist()
        labels = variable_names if layer_idx == 0 else [f"h{layer_idx}_{idx}" for idx in range(len(coeffs))]
        summary = ", ".join(
            f"{label}: a={coeff:.4f}, b={exp:.4f}"
            for label, coeff, exp in zip(labels, coeffs, exponents)
        )
        logging.info(f"   Coeff layer {layer_idx}: {summary}")


def run_problem(problem: ProblemSpec, args: argparse.Namespace, device: torch.device) -> BenchmarkResult:
    operator_names = infer_operator_names(problem.formula, args.operator_mode, args.alternating_axb, args.tree_layout)
    search_settings = resolve_search_settings(args)
    logging.info(f"\n🧪 {problem.problem_id}: {problem.name}")
    logging.info(f"   Formula: {problem.formula}")
    logging.info(f"   Operators: {operator_names}")
    logging.info(f"   Layout: {args.tree_layout}")
    logging.info(f"   Alternating a*x^b: {'on' if args.alternating_axb else 'off'}")
    logging.info(f"   Constant leaf: {'on' if args.use_constant_input else 'off'}")
    if should_use_operator_curriculum(len(problem.variable_names), operator_names, search_settings):
        logging.info(f"   Operator curriculum: on ({search_settings.operator_curriculum_epochs} warmup epochs)")
    elif search_settings.use_operator_curriculum:
        logging.info("   Operator curriculum: skipped for this problem")
    if search_settings.complexity_weight > 0.0:
        logging.info(f"   Complexity weight: {search_settings.complexity_weight:.4f}")

    x_train, y_train, x_val, y_val, x_test, y_test = sample_problem_data(
        problem,
        train_size=args.samples_train,
        val_size=args.samples_val,
        test_size=args.samples_test,
        seed=args.seed,
        device=device,
    )

    if search_settings.use_candidate_ranking:
        candidates: list[CandidateResult] = []
        for attempt_index in range(args.attempts):
            candidate = train_candidate(
                problem=problem,
                args=args,
                device=device,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                operator_names=operator_names,
                search_settings=search_settings,
                attempt_index=attempt_index,
            )
            logging.info(
                "   Candidate %d/%d: val=%.4f test=%.4f complexity=%.2f score=%.4f ops=%s",
                attempt_index + 1,
                args.attempts,
                candidate.validation_r2,
                candidate.test_r2,
                candidate.model_complexity,
                candidate.selection_score,
                candidate.selected_operators,
            )
            candidates.append(candidate)
        best_candidate = choose_best_candidate(candidates, search_settings)
        best_model = best_candidate.model
        best_val_r2 = best_candidate.validation_r2
        test_r2 = best_candidate.test_r2
        selected_operators = best_candidate.selected_operators
        model_complexity = best_candidate.model_complexity
        selection_score = best_candidate.selection_score
    else:
        best_candidate = train_candidate(
            problem=problem,
            args=SimpleNamespace(**{**vars(args), "attempts": 1}),
            device=device,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            operator_names=operator_names,
            search_settings=search_settings,
            attempt_index=0,
        )
        best_model = best_candidate.model
        best_val_r2 = best_candidate.validation_r2
        test_r2 = best_candidate.test_r2
        selected_operators = best_candidate.selected_operators
        model_complexity = best_candidate.model_complexity
        selection_score = best_candidate.selection_score

    success = test_r2 >= args.success_threshold
    logging.info(f"   Validation R²: {best_val_r2:.4f}")
    logging.info(f"   Test R²:       {test_r2:.4f}")
    logging.info(f"   Complexity:    {model_complexity:.2f}")
    logging.info(f"   Select score:  {selection_score:.4f}")
    logging.info(f"   Chosen ops:    {selected_operators}")
    logging.info(f"   Success:       {'yes' if success else 'no'}")

    return BenchmarkResult(
        problem_id=problem.problem_id,
        name=problem.name,
        validation_r2=float(best_val_r2),
        test_r2=float(test_r2),
        operator_set=operator_names,
        selected_operators=selected_operators,
        alternating_axb=args.alternating_axb,
        model_complexity=float(model_complexity),
        selection_score=float(selection_score),
        success=success,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    logging.info(f"🖥️  Using device: {device}")
    all_specs = load_problem_specs(args.dataset)
    selected_specs = select_problems(all_specs, args.problems, args.all, args.max_problems)
    logging.info(f"📚 Running {len(selected_specs)} problem(s) from {args.dataset}")

    results: list[BenchmarkResult] = []
    failures = 0
    for index, problem in enumerate(selected_specs, start=1):
        logging.info(f"\n===== [{index}/{len(selected_specs)}] {problem.problem_id} =====")
        try:
            results.append(run_problem(problem, args, device))
        except Exception as exc:
            failures += 1
            logging.exception(f"   ❌ Failed: {problem.problem_id}")
            results.append(
                BenchmarkResult(
                    problem_id=problem.problem_id,
                    name=problem.name,
                    validation_r2=float("nan"),
                    test_r2=float("nan"),
                    operator_set=[],
                    selected_operators=[],
                    alternating_axb=args.alternating_axb,
                    model_complexity=float("nan"),
                    selection_score=float("nan"),
                    success=False,
                    error=str(exc),
                )
            )
            if args.stop_on_error:
                raise

    logging.info("\n📊 Summary")
    if results:
        completed = [item for item in results if item.error is None]
        mean_test_r2 = sum(item.test_r2 for item in completed) / len(completed) if completed else float("nan")
        successes = sum(1 for item in completed if item.success)
        logging.info(f"   Recovered: {successes}/{len(completed)} completed")
        logging.info(f"   Failures:  {failures}")
        if completed:
            logging.info(f"   Mean test R²: {mean_test_r2:.4f}")

        for item in results:
            if item.error is not None:
                logging.info(f"   FAIL {item.problem_id}: {item.error}")
            else:
                status = "PASS" if item.success else "MISS"
                logging.info(
                    f"   {status:4s} {item.problem_id:12s} val={item.validation_r2:.4f} test={item.test_r2:.4f} "
                    f"complexity={item.model_complexity:.2f} score={item.selection_score:.4f}"
                )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "success_threshold": args.success_threshold,
            "results": [asdict(result) for result in results],
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info(f"💾 Wrote summary to {args.output_json}")


if __name__ == "__main__":
    main()