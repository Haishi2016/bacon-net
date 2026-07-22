"""
Fixed (human-authored) BACON logic trees over a concept layer.

Each digit is described by a *human-defined* boolean-style formula over named
concepts, e.g.::

    "loop_upper AND loop_lower AND NOT horizontal_middle"
    "horizontal_top AND (diagonal OR vertical_line) AND NOT loop_upper"

The formula is parsed into an AST and evaluated with BACON's graded-logic
power-mean aggregator (``bacon.aggregators.lsp.full_weight``).  AND / OR are
just the same soft aggregator with a different *andness*, and NOT is the graded
complement ``1 - x``.  The structure is FROZEN (human-defined); only the CNN
that produces the concept probabilities is trained.  Because the label loss is
back-propagated through this fixed symbolic structure, it acts as *structural
reinforcement*: the CNN is pushed to extract concepts that make the human logic
true for the correct class and false for the others.

This module has no trainable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torch.nn as nn

from bacon.aggregators.lsp.full_weight import lsp_power_mean


# --------------------------------------------------------------------------- #
# AST node types
# --------------------------------------------------------------------------- #
@dataclass
class Concept:
    name: str


@dataclass
class Not:
    child: "Node"


@dataclass
class And:
    children: List["Node"]


@dataclass
class Or:
    children: List["Node"]


Node = Union[Concept, Not, And, Or]


# --------------------------------------------------------------------------- #
# Tokenizer + recursive-descent parser for the little AND/OR/NOT DSL
# --------------------------------------------------------------------------- #
_KEYWORDS = {"and", "or", "not"}


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    buf = ""
    for ch in text:
        if ch in "()":
            if buf.strip():
                tokens.append(buf.strip())
            buf = ""
            tokens.append(ch)
        elif ch.isspace():
            if buf.strip():
                tokens.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        tokens.append(buf.strip())
    return tokens


class _Parser:
    """Grammar (lowest to highest precedence): OR -> AND -> NOT -> atom."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self):
        tok = self._peek()
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self._parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token '{self._peek()}' in formula")
        return node

    def _parse_or(self) -> Node:
        children = [self._parse_and()]
        while (tok := self._peek()) is not None and tok.lower() == "or":
            self._next()
            children.append(self._parse_and())
        return children[0] if len(children) == 1 else Or(children)

    def _parse_and(self) -> Node:
        children = [self._parse_not()]
        while (tok := self._peek()) is not None and tok.lower() == "and":
            self._next()
            children.append(self._parse_not())
        return children[0] if len(children) == 1 else And(children)

    def _parse_not(self) -> Node:
        tok = self._peek()
        if tok is not None and tok.lower() == "not":
            self._next()
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> Node:
        tok = self._next()
        if tok is None:
            raise ValueError("Unexpected end of formula")
        if tok == "(":
            node = self._parse_or()
            if self._next() != ")":
                raise ValueError("Missing closing parenthesis")
            return node
        if tok.lower() in _KEYWORDS:
            raise ValueError(f"Unexpected keyword '{tok}'")
        return Concept(tok)


def parse_formula(text: str) -> Node:
    return _Parser(_tokenize(text)).parse()


def collect_concepts(node: Node, out: set) -> set:
    if isinstance(node, Concept):
        out.add(node.name)
    elif isinstance(node, Not):
        collect_concepts(node.child, out)
    else:  # And / Or
        for c in node.children:
            collect_concepts(c, out)
    return out


# --------------------------------------------------------------------------- #
# Torch module: a bank of fixed BACON logic trees (one per class)
# --------------------------------------------------------------------------- #
class BaconLogicBank(nn.Module):
    """Evaluate one fixed BACON tree per class over a shared concept vector.

    Args:
        concept_names: ordered list mapping concept index -> name.
        formulas: dict class_index -> formula string.
        and_andness: graded-logic andness for AND nodes (1.0 = strong AND).
        or_andness:  graded-logic andness for OR nodes (0.0 = strong OR).
    """

    def __init__(
        self,
        concept_names: List[str],
        formulas: Dict[int, str],
        and_andness: float = 1.0,
        or_andness: float = 0.0,
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.concept_index = {n: i for i, n in enumerate(self.concept_names)}
        self.and_andness = float(and_andness)
        self.or_andness = float(or_andness)

        self.num_classes = len(formulas)
        self.asts: List[Node] = []
        for k in range(self.num_classes):
            if k not in formulas:
                raise ValueError(f"Missing formula for class {k}")
            ast = parse_formula(formulas[k])
            used = collect_concepts(ast, set())
            unknown = used - set(self.concept_index)
            if unknown:
                raise ValueError(
                    f"Class {k} references unknown concepts {sorted(unknown)}"
                )
            self.asts.append(ast)

    # -- evaluation ------------------------------------------------------- #
    def _eval(self, node: Node, probs: torch.Tensor) -> torch.Tensor:
        """probs: (B, n_concepts) in [0,1]. Returns (B,) truth in (0,1)."""
        if isinstance(node, Concept):
            return probs[:, self.concept_index[node.name]]
        if isinstance(node, Not):
            return 1.0 - self._eval(node.child, probs)

        # And / Or -> graded power-mean aggregation with equal weights.
        children = [self._eval(c, probs) for c in node.children]
        X = torch.stack(children, dim=0)  # (N, B)
        n = X.shape[0]
        w = torch.full((n, 1), 1.0 / n, dtype=X.dtype, device=X.device)
        a = self.and_andness if isinstance(node, And) else self.or_andness
        return lsp_power_mean(X, a, w, eps=1e-6)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """probs: (B, n_concepts) -> class truths (B, num_classes) in (0,1)."""
        truths = [self._eval(self.asts[k], probs) for k in range(self.num_classes)]
        return torch.stack(truths, dim=1)

    # -- introspection ---------------------------------------------------- #
    def ideal_concept_matrix(self) -> torch.Tensor:
        """For each class, the idealized concept vector implied by its formula.

        1.0 for a positive literal, 0.0 for a negated literal, 0.5 for
        don't-care.  Used only to check that the human rule set yields
        distinct per-class targets (separability sanity check).
        """
        M = torch.full((self.num_classes, len(self.concept_names)), 0.5)
        for k, ast in enumerate(self.asts):
            self._fill_ideal(ast, M[k], positive=True)
        return M

    def _fill_ideal(self, node: Node, row: torch.Tensor, positive: bool):
        if isinstance(node, Concept):
            row[self.concept_index[node.name]] = 1.0 if positive else 0.0
        elif isinstance(node, Not):
            self._fill_ideal(node.child, row, not positive)
        else:
            for c in node.children:
                self._fill_ideal(c, row, positive)
