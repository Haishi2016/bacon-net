#!/usr/bin/env python3
"""Group Feynman benchmark results by formula family for triage."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import sympy

from samples.feynman.main import FUNCTION_NAME_MAP, build_formula_locals, load_problem_specs, parse_formula_expression


SYMBOL_PATTERN = re.compile(r"\b[a-zA-Z_]\w*\b")
FUNCTION_NAMES = {
    "exp", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "log", "ln",
}


@dataclass
class ResultRow:
    problem_id: str
    name: str
    formula: str
    test_r2: float | None
    validation_r2: float | None
    success: bool
    error: str | None
    group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze grouped Feynman benchmark results.")
    parser.add_argument("--results-json", type=Path, required=True, help="Path to a JSON summary produced by samples/feynman/main.py")
    parser.add_argument("--dataset", choices=["feynman", "bonus"], default="feynman")
    parser.add_argument("--top-k", type=int, default=5, help="Number of best/worst/near-miss examples to print per summary section")
    parser.add_argument("--write-shortlist", type=Path, default=None, help="Optional path to write a markdown triage shortlist")
    return parser.parse_args()


def normalize_formula(formula: str) -> str:
    normalized = formula.strip()
    for source_name, target_name in FUNCTION_NAME_MAP.items():
        normalized = normalized.replace(source_name, target_name)
    return normalized


def is_sum_of_pairwise_products(expr: sympy.Expr) -> bool:
    terms = sympy.Add.make_args(sympy.expand(expr))
    if len(terms) < 2:
        return False
    for term in terms:
        powers = sympy.Poly(term).total_degree() if term.free_symbols else 0
        if powers != 2:
            return False
        if len(term.free_symbols) > 2:
            return False
    return True


def classify_formula(formula: str) -> str:
    normalized_formula = normalize_formula(formula)
    try:
        expr = parse_formula_expression(normalized_formula)
    except Exception:
        lower_formula = normalized_formula.lower()
        if "exp(" in lower_formula:
            return "exponential"
        if any(token in lower_formula for token in ("asin(", "acos(", "atan(")):
            return "inverse-trig"
        if any(token in lower_formula for token in ("sin(", "cos(", "tan(", "sinh(", "cosh(", "tanh(")):
            return "trig"
        if "sqrt(" in lower_formula:
            return "sqrt-radical"
        if "/" in lower_formula:
            return "rational"
        if "+" in lower_formula or "-" in lower_formula[1:]:
            return "additive-polynomial"
        return "other"
    func_names = {func.func.__name__.lower() for func in expr.atoms(sympy.Function)}

    if "exp" in func_names:
        return "exponential"
    if func_names & {"asin", "acos", "atan"}:
        return "inverse-trig"
    if func_names & {"sin", "cos", "tan"}:
        return "trig"
    if expr.has(sympy.sqrt) or any(atom.is_Pow and atom.exp == sympy.Rational(1, 2) for atom in expr.atoms(sympy.Pow)):
        return "sqrt-radical"
    if is_sum_of_pairwise_products(expr):
        return "sum-of-products"

    expanded = sympy.expand(expr)
    numerator, denominator = sympy.fraction(sympy.together(expanded))
    if denominator != 1:
        return "rational"
    if expanded.is_Mul:
        return "pure-product"
    if expanded.is_Add:
        return "additive-polynomial"
    return "other"


def load_rows(results_json: Path, dataset: str) -> list[ResultRow]:
    results_payload = json.loads(results_json.read_text(encoding="utf-8"))
    specs = {spec.problem_id: spec for spec in load_problem_specs(dataset)}
    rows: list[ResultRow] = []

    for item in results_payload["results"]:
        spec = specs.get(item["problem_id"])
        formula = spec.formula if spec is not None else ""
        test_r2 = item.get("test_r2")
        validation_r2 = item.get("validation_r2")
        if isinstance(test_r2, float) and math.isnan(test_r2):
            test_r2 = None
        if isinstance(validation_r2, float) and math.isnan(validation_r2):
            validation_r2 = None
        rows.append(
            ResultRow(
                problem_id=item["problem_id"],
                name=item.get("name", item["problem_id"]),
                formula=formula,
                test_r2=test_r2,
                validation_r2=validation_r2,
                success=bool(item.get("success", False)),
                error=item.get("error"),
                group=classify_formula(formula) if formula else "unknown",
            )
        )
    return rows


def format_r2(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.4f}"


def build_shortlist_markdown(rows: list[ResultRow], grouped: dict[str, list[ResultRow]]) -> str:
    completed_rows = [row for row in rows if row.error is None and row.test_r2 is not None]
    rational_near_misses = sorted(
        [row for row in completed_rows if row.group == "rational" and not row.success and row.test_r2 is not None and row.test_r2 >= 0.8],
        key=lambda row: row.test_r2 if row.test_r2 is not None else -float("inf"),
        reverse=True,
    )
    hard_failures = [row for row in rows if row.error is not None]
    trig_candidates = sorted(
        [row for row in completed_rows if row.group == "trig" and not row.success],
        key=lambda row: row.test_r2 if row.test_r2 is not None else -float("inf"),
        reverse=True,
    )
    exponential_candidates = sorted(
        [row for row in completed_rows if row.group == "exponential" and not row.success],
        key=lambda row: row.test_r2 if row.test_r2 is not None else -float("inf"),
        reverse=True,
    )

    lines = [
        "# Feynman Triage Shortlist",
        "",
        "## Priority 1: Hard Failures",
    ]
    for row in hard_failures:
        lines.append(f"- {row.problem_id} [{row.group}] error={row.error} formula={row.formula}")

    lines.extend([
        "",
        "## Priority 2: Rational Near-Misses",
    ])
    for row in rational_near_misses:
        lines.append(f"- {row.problem_id} test={format_r2(row.test_r2)} val={format_r2(row.validation_r2)} formula={row.formula}")

    lines.extend([
        "",
        "## Priority 3: Trig Candidates",
    ])
    for row in trig_candidates[:10]:
        lines.append(f"- {row.problem_id} test={format_r2(row.test_r2)} val={format_r2(row.validation_r2)} formula={row.formula}")

    lines.extend([
        "",
        "## Priority 4: Exponential Candidates",
    ])
    for row in exponential_candidates[:10]:
        lines.append(f"- {row.problem_id} test={format_r2(row.test_r2)} val={format_r2(row.validation_r2)} formula={row.formula}")

    lines.extend([
        "",
        "## Group Snapshot",
    ])
    for group_name, items in sorted(grouped.items(), key=lambda entry: (-len(entry[1]), entry[0])):
        completed = [item for item in items if item.error is None and item.test_r2 is not None]
        successes = sum(1 for item in completed if item.success)
        mean_test_r2 = sum(item.test_r2 for item in completed) / len(completed) if completed else float("nan")
        lines.append(
            f"- {group_name}: total={len(items)} completed={len(completed)} pass={successes} mean_test_r2={mean_test_r2:.4f}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_json, args.dataset)

    grouped: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[row.group].append(row)

    print("Grouped Summary")
    for group_name, items in sorted(grouped.items(), key=lambda entry: (-len(entry[1]), entry[0])):
        completed = [item for item in items if item.error is None and item.test_r2 is not None]
        successes = sum(1 for item in completed if item.success)
        mean_test_r2 = sum(item.test_r2 for item in completed) / len(completed) if completed else float("nan")
        print(f"{group_name}: total={len(items)} completed={len(completed)} pass={successes} pass%={(100.0 * successes / len(items)) if items else float('nan'):.1f} mean_test_r2={mean_test_r2:.4f}")

    completed_rows = [row for row in rows if row.error is None and row.test_r2 is not None]
    ranked = sorted(completed_rows, key=lambda row: row.test_r2 if row.test_r2 is not None else -float("inf"), reverse=True)
    near_misses = [row for row in ranked if not row.success and row.test_r2 is not None and row.test_r2 >= 0.8]
    failures = [row for row in rows if row.error is not None]

    print("\nTop Results")
    for row in ranked[:args.top_k]:
        print(f"{row.problem_id}: group={row.group} test={format_r2(row.test_r2)} formula={row.formula}")

    print("\nNear Misses")
    for row in near_misses[:args.top_k]:
        print(f"{row.problem_id}: group={row.group} test={format_r2(row.test_r2)} formula={row.formula}")

    print("\nHard Failures")
    for row in failures[:args.top_k]:
        print(f"{row.problem_id}: group={row.group} error={row.error} formula={row.formula}")

    print("\nWorst Completed")
    for row in ranked[-args.top_k:]:
        print(f"{row.problem_id}: group={row.group} test={format_r2(row.test_r2)} formula={row.formula}")

    if args.write_shortlist is not None:
        args.write_shortlist.parent.mkdir(parents=True, exist_ok=True)
        args.write_shortlist.write_text(build_shortlist_markdown(rows, grouped), encoding="utf-8")
        print(f"\nWrote shortlist to {args.write_shortlist}")


if __name__ == "__main__":
    main()