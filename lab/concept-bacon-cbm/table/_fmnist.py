"""
Shared helpers for the FashionMNIST table cells (iFMNIST / cFMNIST).

Every cell reuses the *verified* CREAM reproduction (cream/run_cream.py +
cream/models.py) so a table cell is guaranteed identical to the reproduction.
Each cell script stays a thin one-model-per-file wrapper: it imports ``run_cell``
and calls it with its (model, dataset).

Model keys -> paper rows:
  blackbox     Black-box
  ctruey       C_true -> Y
  cbm          CBM              (SoftCBM: independent sigmoid concepts -> linear)
  cbm+sc       CBM+SC           (SoftCBM + regularized side-channel)
  cream-wo-sc  CREAM w/o SC     (CREAM, side-channel OFF at eval)
  cream        CREAM            (CREAM, side-channel ON)
"""

from __future__ import annotations

import os

import _bench                                                   # sets sys.path
from run_cream import (evaluate, eval_ctruey, load_fmnist,      # noqa: E402
                       train_ctruey, train_model)
import fmnist_concepts as fc                                    # noqa: E402
from models import BlackBox, CREAM, SoftCBM, SoftCBMSC, BaconCBM  # noqa: E402

_DATA = os.path.join(_bench._REPO, "benchmarks", "mnist-addition", "data")
SPECS = {"ifmnist": fc.IFMNIST, "cfmnist": fc.CFMNIST, "sfmnist": fc.SFMNIST}

# model_key -> (builder, has_concepts, has_side_channel)
_MODELS = {
    "blackbox":    (lambda s: BlackBox(),                 False, False),
    "cbm":         (lambda s: SoftCBM(s),                 True,  False),
    "cbm+sc":      (lambda s: SoftCBMSC(s, dropout_p=0.9), True, True),
    "cream-wo-sc": (lambda s: CREAM(s, dropout_p=0.9),    True,  True),
    "cream":       (lambda s: CREAM(s, dropout_p=0.9),    True,  True),
    # OCBM = our BACON-CBM: fixed logic trees (v1/v2) or data-fine-tuned (v3).
    "ocbm":        (lambda s: BaconCBM(s),                     True, False),
    "ocbm-ft":     (lambda s: BaconCBM(s, finetune_logic=True), True, False),
}

# pretty labels for the report header
LABELS = {
    "blackbox": "Black-box", "ctruey": "C_true->Y", "cbm": "CBM",
    "cbm+sc": "CBM+SC", "cream-wo-sc": "CREAM w/o SC", "cream": "CREAM",
    "ocbm": "OCBM", "ocbm-ft": "OCBM (fine-tuned)",
}


def run_cell(model_key: str, dataset: str, iters: int,
             epochs: int = 20, batch_size: int = 256, lam: float = 0.3,
             seed: int = 0):
    """Train ``model_key`` on ``dataset`` for ``iters`` seeds; return (acc_y, acc_c).

    acc_c is None for rows without a measurable concept accuracy (blackbox, ctruey).
    """
    spec = SPECS[dataset]
    device = _bench.get_device()
    tl, vl = load_fmnist(_DATA, batch_size)
    label = LABELS.get(model_key, model_key)
    print(f"{label} / {dataset}   iters={iters}  epochs={epochs}  "
          f"K={spec.K}  device={device}")

    acc_y, acc_c = [], []
    for it in range(iters):
        _bench.set_seed(seed + it)
        if model_key == "ctruey":
            m = train_ctruey(spec, tl, device)
            ay = eval_ctruey(m, vl, spec, device)
            ac = None
        else:
            build, has_c, has_side = _MODELS[model_key]
            model = build(spec)
            train_model(model, tl, spec, device, epochs, lam, has_c)
            full, cpath, cacc = evaluate(model, vl, spec, device, has_side)
            ay = cpath if model_key == "cream-wo-sc" else full
            ac = cacc if has_c else None
        acc_y.append(ay)
        if ac is not None:
            acc_c.append(ac)
        cstr = f"  ACC_C = {ac * 100:.2f}" if ac is not None else ""
        print(f"  [iter {it + 1}/{iters}]  ACC_Y = {ay * 100:.2f}{cstr}")

    return acc_y, (acc_c or None)


def run_gl_cell(trees, dataset: str, iters: int, trainable: bool = False,
                epochs: int = 20, batch_size: int = 256, lam: float = 1.0, seed: int = 0):
    """OCBM GL-tree cell: JSON trees -> GLTreeCBM (spec activation) -> ACC_Y/ACC_C."""
    import _gltree
    spec = SPECS[dataset]
    device = _bench.get_device()
    tl, vl = load_fmnist(_DATA, batch_size)
    print(f"OCBM GL ({'ft' if trainable else 'fixed'}) / {dataset}   iters={iters}  "
          f"epochs={epochs}  K={spec.K}  device={device}")
    acc_y, acc_c = [], []
    for it in range(iters):
        _bench.set_seed(seed + it)
        model = _gltree.GLTreeCBM(spec.concept_names, trees, spec=spec,
                                  feat_dim=128, trainable=trainable)
        train_model(model, tl, spec, device, epochs, lam, True)
        full, _cpath, cacc = evaluate(model, vl, spec, device, False)
        acc_y.append(full)
        acc_c.append(cacc)
        print(f"  [iter {it + 1}/{iters}]  ACC_Y = {full * 100:.2f}  ACC_C = {cacc * 100:.2f}")
    return acc_y, acc_c
