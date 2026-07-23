"""
Trainable fixed-structure logic trees (graded-logic fine-tuning).

This module lets a *human-defined* BACON tree be fine-tuned with gradient
descent while keeping its **structure frozen**.  The topology -- which inputs
feed which node, and whether each node is a conjunction / disjunction /
negation -- is fixed at construction.  Training only adjusts, per internal node:

  * the **andness** of the aggregator (via the generic GL anchor mixture), and
  * the **per-input weights** (importance of each child).

Negation (``NOT``) is fixed as the graded complement ``1 - x``.

Each conjunction / disjunction node is backed by a
:class:`~bacon.aggregators.lsp.generic_gl.GenericGLAggregator` (``weight_mode=
'static'``), whose learnable anchor weights move the node continuously along the
``min -> ... -> max`` andness continuum.  Per-input weights are applied in the
standard graded-logic way: a child with weight ``w`` is blended toward the
node's neutral element (``w * x + (1 - w) * neutral``), so a low weight makes an
input irrelevant to that node.

Trees are defined either from a small formula DSL::

    "loop_upper AND loop_lower AND NOT horizontal_middle"
    "top AND (diagonal OR vertical) AND NOT loop"

or by building the AST nodes directly.  A :class:`FixedGLTreeBank` evaluates one
tree per class over a shared input vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bacon.aggregators.lsp.generic_gl import GenericGLAggregator, ANCHOR_ANDNESS
from bacon.aggregators.lsp.full_weight import lsp_power_mean

_DEFAULT_ANCHORS = ("min", "harmonic", "geometric", "mean", "quadratic", "max")
_AGGREGATORS = ("gl.generic", "lsp.full_weight")


def _andness_to_a(target: float) -> float:
    """Inverse of ``a = 3*sigmoid(p) - 1`` for the lsp full_weight andness in [-1,2]."""
    import math
    s = min(max((target + 1.0) / 3.0, 1e-4), 1 - 1e-4)
    return math.log(s / (1.0 - s))


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #
@dataclass
class Leaf:
    name: str


@dataclass
class Not:
    child: "Node"


@dataclass
class And:
    children: List["Node"]
    node_id: int = field(default=-1)      # index into the module's aggregator bank


@dataclass
class Or:
    children: List["Node"]
    node_id: int = field(default=-1)


Node = Union[Leaf, Not, And, Or]


# --------------------------------------------------------------------------- #
# DSL parser (AND / OR / NOT / parentheses)
# --------------------------------------------------------------------------- #
_KEYWORDS = {"and", "or", "not"}


def _tokenize(text: str) -> List[str]:
    tokens, buf = [], ""
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
    def __init__(self, tokens: List[str]):
        self.tokens, self.pos = tokens, 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self):
        tok = self._peek()
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self._parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(f"unexpected token '{self._peek()}'")
        return node

    def _parse_or(self) -> Node:
        children = [self._parse_and()]
        while (t := self._peek()) is not None and t.lower() == "or":
            self._next()
            children.append(self._parse_and())
        return children[0] if len(children) == 1 else Or(children)

    def _parse_and(self) -> Node:
        children = [self._parse_not()]
        while (t := self._peek()) is not None and t.lower() == "and":
            self._next()
            children.append(self._parse_not())
        return children[0] if len(children) == 1 else And(children)

    def _parse_not(self) -> Node:
        t = self._peek()
        if t is not None and t.lower() == "not":
            self._next()
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> Node:
        t = self._next()
        if t is None:
            raise ValueError("unexpected end of formula")
        if t == "(":
            node = self._parse_or()
            if self._next() != ")":
                raise ValueError("missing closing parenthesis")
            return node
        if t.lower() in _KEYWORDS:
            raise ValueError(f"unexpected keyword '{t}'")
        return Leaf(t)


def parse_formula(text: str) -> Node:
    return _Parser(_tokenize(text)).parse()


def _collect_leaves(node: Node, out: set) -> set:
    if isinstance(node, Leaf):
        out.add(node.name)
    elif isinstance(node, Not):
        _collect_leaves(node.child, out)
    else:
        for c in node.children:
            _collect_leaves(c, out)
    return out


# --------------------------------------------------------------------------- #
# Trainable fixed-structure tree
# --------------------------------------------------------------------------- #
class FixedGLTree(nn.Module):
    """A single fixed-structure logic tree with trainable graded-logic nodes.

    The tree TOPOLOGY is fixed; training adjusts each node's andness + per-input
    weights.  Two aggregator backends are available:

      * ``"gl.generic"``   -- generic GL aggregator; andness = a learnable convex
        mixture over anchor operators (min..max); per-input weights blend each
        child toward the node's neutral element.
      * ``"lsp.full_weight"`` -- BACON's weighted power-mean aggregator; andness
        is a single learnable scalar in ``[-1, 2]`` and per-input weights are a
        learnable convex weighting (softmax) applied natively by the aggregator.

    Args:
        formula: formula string (DSL) or a pre-built AST :class:`Node`.
        input_names: ordered names mapping input index -> leaf name.
        aggregator: ``"gl.generic"`` (default) or ``"lsp.full_weight"``.
        anchors: GL anchor operators (``gl.generic`` only).
        and_init / or_init: initial andness for AND / OR nodes in ``[0, 1]``
            (1 = strong AND, 0 = strong OR).  Trainable afterwards.
        learn_weights: if True, per-input weights are trainable; else fixed.
        init_weight: initial per-input weight in ``(0, 1)`` (``gl.generic`` only).
        tau: GL softmax temperature over anchors (``gl.generic`` only).
    """

    def __init__(
        self,
        formula: Union[str, Node],
        input_names: Sequence[str],
        aggregator: str = "gl.generic",
        anchors: Sequence[str] = _DEFAULT_ANCHORS,
        and_init: float = 0.75,
        or_init: float = 0.25,
        learn_weights: bool = True,
        init_weight: float = 0.9,
        tau: float = 0.5,
    ):
        super().__init__()
        if aggregator not in _AGGREGATORS:
            raise ValueError(f"aggregator must be one of {_AGGREGATORS}")
        self.input_names = list(input_names)
        self.input_index = {n: i for i, n in enumerate(self.input_names)}
        self.aggregator = aggregator
        self.anchors = tuple(anchors)
        self._andness_vec = torch.tensor([ANCHOR_ANDNESS[a] for a in self.anchors])

        self.ast = parse_formula(formula) if isinstance(formula, str) else formula
        unknown = _collect_leaves(self.ast, set()) - set(self.input_index)
        if unknown:
            raise ValueError(f"formula references unknown inputs {sorted(unknown)}")

        self.aggregators = nn.ModuleList()        # gl.generic backend
        self.andness_logits = nn.ParameterList()  # lsp.full_weight backend
        self.weight_logits = nn.ParameterList()   # both backends
        self._assign(self.ast, and_init, or_init, learn_weights, init_weight, tau)

    # -- construction ---------------------------------------------------- #
    def _assign(self, node, and_init, or_init, learn_weights, init_weight, tau):
        if isinstance(node, Leaf):
            return
        if isinstance(node, Not):
            self._assign(node.child, and_init, or_init, learn_weights, init_weight, tau)
            return
        # And / Or
        for c in node.children:
            self._assign(c, and_init, or_init, learn_weights, init_weight, tau)
        target = and_init if isinstance(node, And) else or_init
        k = len(node.children)
        if self.aggregator == "gl.generic":
            agg = GenericGLAggregator(anchors=self.anchors, weight_mode="static", tau=tau)
            # concentrate the anchor mixture on operators whose andness is near target
            with torch.no_grad():
                agg.alpha_logits.copy_(-8.0 * (self._andness_vec - target) ** 2)
            w0 = torch.logit(torch.tensor(float(init_weight)))
            w = nn.Parameter(torch.full((k,), w0), requires_grad=learn_weights)
            node.node_id = len(self.aggregators)
            self.aggregators.append(agg)
            self.weight_logits.append(w)
        else:  # lsp.full_weight -- native scalar andness + convex input weights
            a = nn.Parameter(torch.tensor(_andness_to_a(target)))
            w = nn.Parameter(torch.zeros(k), requires_grad=learn_weights)   # uniform
            node.node_id = len(self.andness_logits)
            self.andness_logits.append(a)
            self.weight_logits.append(w)

    # -- evaluation ------------------------------------------------------ #
    def _node_neutral(self, agg: GenericGLAggregator) -> torch.Tensor:
        """Differentiable neutral element = the node's effective andness in [0,1]."""
        w = F.softmax(agg.alpha_logits / agg.tau, dim=0)
        a = self._andness_vec.to(w.device, w.dtype)
        return (w * a).sum().clamp(0.0, 1.0)

    def _node_andness(self, node_id: int) -> torch.Tensor:
        """lsp.full_weight scalar andness a = 3*sigmoid(p) - 1 in (-1, 2)."""
        return 3.0 * torch.sigmoid(self.andness_logits[node_id]) - 1.0

    def _eval(self, node: Node, x: torch.Tensor) -> torch.Tensor:
        if isinstance(node, Leaf):
            return x[:, self.input_index[node.name]]
        if isinstance(node, Not):
            return 1.0 - self._eval(node.child, x)
        # And / Or
        children = [self._eval(c, x) for c in node.children]
        xs = torch.stack(children, dim=0)                      # (k, B)
        if self.aggregator == "gl.generic":
            agg = self.aggregators[node.node_id]
            w = torch.sigmoid(self.weight_logits[node.node_id]).unsqueeze(-1)   # (k,1)
            neutral = self._node_neutral(agg)
            xs = w * xs + (1.0 - w) * neutral                  # blend-to-neutral weighting
            return agg.forward(xs)                             # (B,)
        # lsp.full_weight: convex weights (sum to 1) + scalar andness, native
        w_norm = F.softmax(self.weight_logits[node.node_id], dim=0).unsqueeze(-1)  # (k,1)
        a = self._node_andness(node.node_id)
        return lsp_power_mean(xs, a, w_norm)                   # (B,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_inputs) in [0,1] -> tree truth (B,) in [0,1]."""
        return self._eval(self.ast, x)

    # -- introspection --------------------------------------------------- #
    def describe(self) -> List[dict]:
        """Per-node andness + input weights (post-order to match node ids)."""
        rows = []

        @torch.no_grad()
        def walk(node):
            if isinstance(node, Leaf):
                return
            if isinstance(node, Not):
                walk(node.child)
                return
            for c in node.children:
                walk(c)
            if self.aggregator == "gl.generic":
                andness = float(self._node_neutral(self.aggregators[node.node_id]))
                w = torch.sigmoid(self.weight_logits[node.node_id]).tolist()
            else:
                andness = float(self._node_andness(node.node_id))
                w = F.softmax(self.weight_logits[node.node_id], dim=0).tolist()
            rows.append({
                "id": node.node_id,
                "op": "AND" if isinstance(node, And) else "OR",
                "andness": andness,
                "weights": [round(v, 3) for v in w],
            })

        walk(self.ast)
        return rows


class FixedGLTreeBank(nn.Module):
    """One trainable fixed-structure tree per class over a shared input vector."""

    def __init__(
        self,
        input_names: Sequence[str],
        formulas: Dict[int, Union[str, Node]],
        **tree_kwargs,
    ):
        super().__init__()
        self.input_names = list(input_names)
        self.num_classes = len(formulas)
        self.trees = nn.ModuleList()
        for k in range(self.num_classes):
            if k not in formulas:
                raise ValueError(f"missing formula for class {k}")
            self.trees.append(FixedGLTree(formulas[k], input_names, **tree_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_inputs) -> class truths (B, num_classes)."""
        return torch.stack([t(x) for t in self.trees], dim=1)

    def describe(self) -> Dict[int, List[dict]]:
        return {k: t.describe() for k, t in enumerate(self.trees)}
