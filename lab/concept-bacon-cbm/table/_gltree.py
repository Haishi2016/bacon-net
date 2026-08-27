"""
JSON Graded-Logic (GL/LSP) tree evaluator for OCBM v1/v3.

Consumes the structured trees produced by OCBM_V1_TREE_PROMPT.md: a per-class
hierarchical tree whose internal nodes carry a GCD operator (-> andness ``a``)
and per-input weights, and whose leaves reference concepts (optionally negated).
Each node is evaluated with the weighted GL power mean ``lsp_power_mean(X, a, w)``
(``a in [-1, 2]``); negation is the graded complement ``1 - x``.

  * trainable=False -> OCBM v1 (fixed andness + weights).
  * trainable=True  -> OCBM v3 (same structure; andness + weights are learned).

Node JSON schema:
  leaf     : {"concept": <name>, "weight": <w>, "negate": <bool>}
  internal : {"op": <GCD code> | "andness": <float>, "weight": <w>,
              "name": <label>, "children": [ ... ]}
(child ``weight`` is that child's importance inside its parent; top node needs none.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import _bench  # noqa: F401  (sets sys.path for bacon + models)
from bacon.aggregators.lsp.full_weight import lsp_power_mean  # noqa: E402
from models import Backbone  # noqa: E402

# GCD operator code -> andness a in [-1, 2] (from the LSP GCD table).
_OP2A = {
    "CC": 2.0, "HHC": 1.6, "CP": 1.25, "C": 1.0,
    "HC+": 0.93, "HC": 0.86, "HC-": 0.79,
    "SC+": 0.71, "SC": 0.64, "SC-": 0.57,
    "A": 0.5,
    "SD-": 0.43, "SD": 0.36, "SD+": 0.29,
    "HD-": 0.21, "HD": 0.14, "HD+": 0.07,
    "D": 0.0, "HHD": -0.6, "DD": -1.0,
}


def _a_to_p(a: float) -> float:
    """Inverse of a = 3*sigmoid(p) - 1, so a trainable node inits at andness a."""
    s = min(max((a + 1.0) / 3.0, 1e-4), 1.0 - 1e-4)
    return math.log(s / (1.0 - s))


class _Leaf(nn.Module):
    def __init__(self, idx: int, negate: bool):
        super().__init__()
        self.idx = int(idx)
        self.negate = bool(negate)

    def forward(self, c):                       # c: (B, K) -> (B,)
        x = c[:, self.idx]
        return (1.0 - x) if self.negate else x


class _Internal(nn.Module):
    def __init__(self, andness, weights, child_mods, trainable):
        super().__init__()
        self.kids = nn.ModuleList(child_mods)
        self.trainable = trainable
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-6)
        if trainable:
            self.a_param = nn.Parameter(torch.tensor(_a_to_p(float(andness))))
            self.w_logit = nn.Parameter(torch.log(w.clamp_min(1e-4)))
        else:
            self.register_buffer("a_const", torch.tensor(float(andness)))
            self.register_buffer("w_const", w)

    def forward(self, c):
        X = torch.stack([m(c) for m in self.kids], dim=0)     # (n, B)
        if self.trainable:
            a = 3.0 * torch.sigmoid(self.a_param) - 1.0
            w = torch.softmax(self.w_logit, dim=0)
        else:
            a, w = self.a_const, self.w_const
        w = w.view(-1, *([1] * (X.dim() - 1)))
        return lsp_power_mean(X, a, w)


def _build(node, cidx, trainable):
    if "concept" in node:
        return _Leaf(cidx[node["concept"]], node.get("negate", False))
    if "andness" in node:
        a = float(node["andness"])
    else:
        a = _OP2A[str(node["op"]).upper()]
    kids = node["children"]
    child_mods = [_build(k, cidx, trainable) for k in kids]
    weights = [float(k.get("weight", 1.0)) for k in kids]
    return _Internal(a, weights, child_mods, trainable)


class GLTreeBank(nn.Module):
    """Evaluate one GL tree per class over a shared concept vector -> (B, L)."""

    def __init__(self, concept_names, trees: dict, trainable: bool = False):
        super().__init__()
        cidx = {n: i for i, n in enumerate(concept_names)}
        self.labels = sorted(
            trees.keys(),
            key=lambda k: int(k) if str(k).lstrip("-").isdigit() else str(k),
        )
        self.banks = nn.ModuleList(
            [_build(trees[l], cidx, trainable) for l in self.labels])

    def forward(self, c):                       # c: (B, K) -> (B, L)
        return torch.stack([b(c).clamp(1e-6, 1.0 - 1e-6) for b in self.banks], dim=1)


class GLTreeCBM(nn.Module):
    """OCBM head: backbone -> concepts -> per-class GL trees -> logits.

    If ``spec`` is given, concepts use the spec's activation (softmax over mutex
    groups + sigmoid over binary concepts); otherwise plain sigmoid.  Pass a
    ResNet via ``backbone`` (with feat_dim=512) for CUB/CelebA.
    """

    def __init__(self, concept_names, trees, spec=None, feat_dim=128, backbone=None,
                 trainable=False, logit_temp=6.0):
        super().__init__()
        self.spec = spec
        self.backbone = backbone if backbone is not None else Backbone(feat_dim)
        self.concept = nn.Linear(feat_dim, len(concept_names))
        self.bank = GLTreeBank(concept_names, trees, trainable)
        self.log_temp = nn.Parameter(torch.tensor(math.log(logit_temp)))

    def forward(self, x, use_side=True, harden=False):
        z = self.concept(self.backbone(x))
        if self.spec is not None:
            from models import activate, harden_concepts
            c = activate(z, self.spec)
            ch = harden_concepts(c, self.spec) if harden else c
        else:
            c = torch.sigmoid(z)
            ch = (c > 0.5).float() if harden else c
        t = self.bank(ch).clamp(1e-6, 1.0 - 1e-6)
        logits = self.log_temp.exp() * (torch.log(t) - torch.log1p(-t))
        return logits, c
