"""
Models for the CREAM reproduction + our BACON-logic comparison, on FashionMNIST.

  BlackBox    : CNN -> 10 classes (task-only reference).
  CtrueY      : linear on GROUND-TRUTH concepts -> classes (leakage reference).
  SoftCBM     : CNN -> soft concepts -> linear -> classes (leakage-prone baseline).
  CREAM       : CNN -> splitter -> (masked C->Y over softmax-mutex concepts)
                + regularized (dropout-p) black-box side-channel.
  BaconCBM    : CNN -> softmax-mutex concepts -> FIXED per-class BACON AND-trees
                (our approach; fully transparent, no learned task head).
  BaconCBM+SC : BaconCBM with the same regularized side-channel as CREAM.

All concept-based models predict concepts with a per-mutex-group softmax, matching
CREAM's handling of mutually-exclusive concepts.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# repo root (for the `bacon` package)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from bacon_logic import BaconLogicBank  # noqa: E402
from bacon.fixedGLTree import FixedGLTreeBank  # noqa: E402


def group_softmax(logits: torch.Tensor, groups) -> torch.Tensor:
    """Softmax within each mutually-exclusive concept group."""
    out = torch.empty_like(logits)
    for g in groups:
        out[:, g] = torch.softmax(logits[:, g], dim=1)
    return out


def activate(logits: torch.Tensor, spec) -> torch.Tensor:
    """Softmax over mutex groups + sigmoid over independent binary concepts."""
    out = torch.empty_like(logits)
    for g in spec.mutex_groups:
        out[:, g] = torch.softmax(logits[:, g], dim=1)
    if spec.binary_concepts:
        idx = spec.binary_concepts
        out[:, idx] = torch.sigmoid(logits[:, idx])
    return out


def harden_concepts(c: torch.Tensor, spec=None) -> torch.Tensor:
    """Binarize soft concepts to their hard decisions: one-hot argmax within each
    mutex group, threshold 0.5 for independent binary concepts.  Feeding these
    through the frozen task head measures how much task signal was carried in the
    *continuous* concept values (soft-vs-hard concept leakage).
    """
    mutex = getattr(spec, "mutex_groups", None) if spec is not None else None
    binc = getattr(spec, "binary_concepts", None) if spec is not None else None
    if not mutex and not binc:
        return (c > 0.5).float()
    out = c.clone()
    for g in (mutex or []):
        sub = out[:, g]
        oneh = torch.zeros_like(sub)
        oneh[torch.arange(sub.shape[0], device=sub.device), sub.argmax(1)] = 1.0
        out[:, g] = oneh
    if binc:
        cols = torch.as_tensor(list(binc), device=out.device, dtype=torch.long)
        out[:, cols] = (out[:, cols] > 0.5).float()
    return out


class MaskedLinear(nn.Module):
    """Linear whose weight is fixed-masked to a 0/1 connectivity pattern (StrNN d=0)."""

    def __init__(self, in_f, out_f, mask: torch.Tensor):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)
        self.register_buffer("mask", mask.float())   # (out_f, in_f)

    def forward(self, x):
        return F.linear(x, self.lin.weight * self.mask, self.lin.bias)


class Backbone(nn.Module):
    """Lightweight FashionMNIST CNN (two conv blocks), as in the CREAM setup."""

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, feat_dim), nn.ReLU())
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.fc(self.conv(x))


class BlackBox(nn.Module):
    def __init__(self, feat_dim=128, n_classes=10, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else Backbone(feat_dim)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x)), None


class CtrueY(nn.Module):
    """Linear classifier on ground-truth concepts (the C_true -> Y reference)."""

    def __init__(self, spec):
        super().__init__()
        self.head = nn.Linear(spec.K, spec.A_Y.shape[0])

    def forward(self, concepts):
        return self.head(concepts), None


class SoftCBM(nn.Module):
    """Vanilla soft CBM: INDEPENDENT sigmoid concepts (the leakage-prone baseline)."""

    def __init__(self, spec, feat_dim=128, backbone=None):
        super().__init__()
        self.spec = spec
        self.backbone = backbone if backbone is not None else Backbone(feat_dim)
        self.concept = nn.Linear(feat_dim, spec.K)
        self.head = nn.Linear(spec.K, spec.A_Y.shape[0])

    def forward(self, x, use_side=True, harden=False):
        c = torch.sigmoid(self.concept(self.backbone(x)))
        ch = harden_concepts(c, self.spec) if harden else c
        return self.head(ch), c


def _drop_channel(z: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Drop the ENTIRE side-channel per sample with prob p (CREAM regularization)."""
    if not training or p <= 0:
        return z
    keep = (torch.rand(z.shape[0], 1, device=z.device) > p).float()
    return z * keep


class SoftCBMSC(nn.Module):
    """CBM + a CREAM-style regularized black-box side-channel (the CBM+SC row).

    Identical to SoftCBM (independent sigmoid concepts -> linear head) plus a
    dropout-p-regularized side-channel added to the logits, so the concept path
    must stand on its own while the side-channel absorbs residual task signal.
    """

    def __init__(self, spec, feat_dim=128, d_y=20, dropout_p=0.9, backbone=None):
        super().__init__()
        self.spec = spec
        self.p = dropout_p
        self.d_y = d_y
        L = spec.A_Y.shape[0]
        self.backbone = backbone if backbone is not None else Backbone(feat_dim)
        self.concept = nn.Linear(feat_dim, spec.K)
        self.head = nn.Linear(spec.K, L)
        self.side_in = nn.Linear(feat_dim, d_y)
        self.side = nn.Linear(d_y, L)

    def forward(self, x, use_side=True, harden=False):
        feat = self.backbone(x)
        c = torch.sigmoid(self.concept(feat))
        ch = harden_concepts(c, self.spec) if harden else c
        logits = self.head(ch)
        z_y = self.side_in(feat)
        z_y = _drop_channel(z_y, self.p, self.training) if use_side else torch.zeros_like(z_y)
        return logits + self.side(z_y), c


class CREAM(nn.Module):
    def __init__(self, spec, feat_dim=128, d_c=7, d_y=20, dropout_p=0.9, backbone=None):
        super().__init__()
        self.spec = spec
        self.p = dropout_p
        self.d_c = d_c
        self.d_y = d_y
        L = spec.A_Y.shape[0]
        self.backbone = backbone if backbone is not None else Backbone(feat_dim)
        self.splitter = nn.Linear(feat_dim, d_c * spec.K + d_y)
        # Concept-Concept block (StrNN d=0): masked linear (d_c*K -> K).
        # A_C = identity + hierarchy cliques (parent<->child within a class).
        A_C = torch.eye(spec.K)
        for parents in spec.class_on.values():
            idx = [spec.index[p] for p in parents]
            for a in idx:
                for b in idx:
                    A_C[a, b] = 1.0
        # M_C = A_C^T (Kron) 1_{1 x d_c}: each concept reads the d_c exogenous
        # dims of its parent concepts.
        M_C = torch.kron(A_C.t().contiguous(), torch.ones(1, d_c))       # (K, d_c*K)
        self.ccb = MaskedLinear(d_c * spec.K, spec.K, M_C)
        # side-channel projection z_Y -> L
        self.side = nn.Linear(d_y, L)
        # Concept-Task block: masked linear [C (K) ; z_Y_hat (L)] -> L,
        # mask = [A_Y (L x K) | I_L].
        mask = torch.cat([spec.A_Y, torch.eye(L)], dim=1)  # (L, K+L)
        self.task = MaskedLinear(spec.K + L, L, mask)

    def forward(self, x, use_side=True, harden=False):
        z = self.splitter(self.backbone(x))
        z_c, z_y = z[:, :self.d_c * self.spec.K], z[:, self.d_c * self.spec.K:]
        c = activate(self.ccb(z_c), self.spec)
        ch = harden_concepts(c, self.spec) if harden else c
        if use_side:
            z_y = _drop_channel(z_y, self.p, self.training)
        else:
            z_y = torch.zeros_like(z_y)
        y_side = self.side(z_y)
        logits = self.task(torch.cat([ch, y_side], dim=1))
        return logits, c


class BaconCBM(nn.Module):
    """Our approach: fixed per-class BACON AND-trees over softmax-mutex concepts."""

    def __init__(self, spec, feat_dim=128, logit_temp=6.0, d_y=0, dropout_p=0.9,
                 finetune_logic=False, finetune_aggregator="gl.generic", backbone=None):
        super().__init__()
        self.spec = spec
        self.feat_dim = feat_dim
        self.dropout_p = dropout_p
        self.finetune_logic = finetune_logic
        self.finetune_aggregator = finetune_aggregator
        self.backbone = backbone if backbone is not None else Backbone(feat_dim)
        self.concept = nn.Linear(feat_dim, spec.K)
        if finetune_logic:
            # SAME fixed structure, but andness + input weights are trainable.
            # aggregator: "gl.generic" (anchor mixture) or "lsp.full_weight"
            # (BACON weighted power-mean; native scalar andness + convex weights).
            self.logic = FixedGLTreeBank(spec.concept_names, spec.formulas,
                                         aggregator=finetune_aggregator,
                                         and_init=0.85, or_init=0.15)
        else:
            self.logic = BaconLogicBank(spec.concept_names, spec.formulas,
                                        and_andness=1.0, or_andness=0.0)
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(logit_temp)))))
        self.d_y = d_y
        L = len(spec.formulas)
        if d_y > 0:                                    # optional CREAM-style side-channel
            self.p = dropout_p
            self.side_in = nn.Linear(feat_dim, d_y)
            self.side = nn.Linear(d_y, L)

    def forward(self, x, use_side=True, harden=False):
        feat = self.backbone(x)
        c = activate(self.concept(feat), self.spec)
        ch = harden_concepts(c, self.spec) if harden else c
        truths = self.logic(ch).clamp(1e-6, 1 - 1e-6)
        logits = self.log_temp.exp() * (torch.log(truths) - torch.log1p(-truths))
        if self.d_y > 0:
            z_y = self.side_in(feat)
            if use_side:
                z_y = _drop_channel(z_y, self.p, self.training)
            else:
                z_y = torch.zeros_like(z_y)
            logits = logits + self.side(z_y)
        return logits, c

    # -- save / reuse the trees + trained extractor --------------------- #
    def save(self, path):
        """Persist the BACON trees (spec + formulas) and trained weights."""
        torch.save({
            "spec": self.spec.to_dict(),
            "state_dict": self.state_dict(),
            "feat_dim": self.feat_dim,
            "d_y": self.d_y,
            "dropout_p": self.dropout_p,
            "finetune_logic": self.finetune_logic,
            "finetune_aggregator": self.finetune_aggregator,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Rebuild a BaconCBM (structure + weights) from a checkpoint."""
        import fmnist_concepts as fc
        ckpt = torch.load(path, map_location=device, weights_only=False)
        spec = fc.ConceptSpec.from_dict(ckpt["spec"])
        model = cls(spec, feat_dim=ckpt["feat_dim"], d_y=ckpt["d_y"],
                    dropout_p=ckpt["dropout_p"],
                    finetune_logic=ckpt.get("finetune_logic", False),
                    finetune_aggregator=ckpt.get("finetune_aggregator", "gl.generic"))
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device), spec


# --------------------------------------------------------------------------- #
# Uniform save / load for ALL model types (used by the cross-model zero-shot).
# --------------------------------------------------------------------------- #
def save_checkpoint(model, path, model_type, spec=None, fmnist_acc=None,
                    d_c=7, d_y=0, dropout_p=0.9):
    d = {"model_type": model_type, "state_dict": model.state_dict(),
         "feat_dim": getattr(getattr(model, "backbone", None), "feat_dim", 128),
         "d_c": d_c, "d_y": d_y, "dropout_p": dropout_p, "fmnist_acc": fmnist_acc}
    if spec is not None:
        d["spec"] = spec.to_dict()
    torch.save(d, path)


def load_checkpoint(path, device="cpu"):
    import fmnist_concepts as fc
    d = torch.load(path, map_location=device, weights_only=False)
    mt, fd = d["model_type"], d.get("feat_dim", 128)
    spec = fc.ConceptSpec.from_dict(d["spec"]) if "spec" in d else None
    if mt == "BlackBox":
        m = BlackBox(fd)
    elif mt == "SoftCBM":
        m = SoftCBM(spec, fd)
    elif mt == "CREAM":
        m = CREAM(spec, fd, d.get("d_c", 7), d.get("d_y", 20), d.get("dropout_p", 0.9))
    elif mt in ("BaconCBM", "BaconCBM+SC"):
        m = BaconCBM(spec, fd, d_y=d.get("d_y", 0), dropout_p=d.get("dropout_p", 0.9))
    else:
        raise ValueError(f"unknown model_type {mt}")
    m.load_state_dict(d["state_dict"])
    return m.to(device), spec, mt, d.get("fmnist_acc")
