"""
Compare the human "critiqued" fixed BACON trees against a FINE-TUNED version.

Both models share the exact same per-class tree STRUCTURE (the human formulas).
  - BaconCBM (fixed)     : andness hard-wired (AND=1, OR=0), equal input weights.
  - BaconCBM (ft, gl.generic)  : SAME structure, trainable andness (anchor mixture)
                                 + input weights (bacon.FixedGLTree, GL backend).
  - BaconCBM (ft, full_weight) : SAME structure, trainable SCALAR andness + convex
                                 input weights via BACON's weighted power-mean.

We train both end-to-end on FashionMNIST (same seed) and compare inference (task
and concept accuracy).  A discriminating test then runs each logic head on the
GROUND-TRUTH concepts: if the fine-tuned trees only reach the Boolean ceiling
there, their extra accuracy on predicted concepts was soft-value leakage; if they
beat it, the learned andness is genuine graded-logic expressiveness.

RESULT: on sFMNIST (complete concepts) fine-tuned reaches 100% on TRUE concepts
= genuine GL expressiveness (learned andness ~0.79-0.86), task(pred) 92.5 < 100
(no leakage). On iFMNIST (incomplete) fine-tuned hits 90.8% on predicted but only
60% (ceiling) on TRUE concepts = the +31pt is soft-value LEAKAGE. So andness itself
is NOT leakage; leakage = task(pred) exceeding the true-concept ceiling.

    python eval_finetune.py --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import fmnist_concepts as fc                       # noqa: E402
import run_cream as rc                             # noqa: E402
from models import BaconCBM                         # noqa: E402


def train_eval(spec, finetune, tl, vl, device, epochs, seed, aggregator="gl.generic"):
    torch.manual_seed(seed)
    model = BaconCBM(spec, finetune_logic=finetune,
                     finetune_aggregator=aggregator).to(device)
    rc.train_model(model, tl, spec, device, epochs, 1.0, True)
    task, _, concept = rc.evaluate(model, vl, spec, device, True)
    return model, task, concept


@torch.no_grad()
def eval_on_true_concepts(model, vl, spec, device):
    """Feed the GROUND-TRUTH concept vectors straight into the (possibly
    fine-tuned) logic head, bypassing the CNN.  If the fine-tuned trees only
    reach the Boolean ceiling here, their extra accuracy on *predicted* concepts
    was soft-value leakage; if they beat the ceiling, the learned andness is
    genuine graded-logic expressiveness."""
    model.eval()
    correct = total = 0
    for _, y in vl:
        y = y.to(device)
        ctrue = spec.concept_targets(y)                 # (B, K) hard 0/1
        truths = model.logic(ctrue)                     # (B, num_classes)
        correct += (truths.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--specs", type=str, default="iFMNIST,sFMNIST")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=str,
                    default=os.path.join(_HERE, "..", "..", "..",
                                         "benchmarks", "mnist-addition", "data"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tl, vl = rc.load_fmnist(args.data, 256)
    print(f"device={device}  epochs={args.epochs}  seed={args.seed}")

    for sname in args.specs.split(","):
        spec = fc.SPECS[sname]
        print(f"\n{'='*66}\n{sname}  (concept-only ceiling reference below)\n{'='*66}")

        fx, t_fx, c_fx = train_eval(spec, False, tl, vl, device, args.epochs, args.seed)
        gl, t_gl, c_gl = train_eval(spec, True,  tl, vl, device, args.epochs, args.seed,
                                    aggregator="gl.generic")
        fw, t_fw, c_fw = train_eval(spec, True,  tl, vl, device, args.epochs, args.seed,
                                    aggregator="lsp.full_weight")

        # discriminating test: run each logic head on the TRUE concepts.
        tt_fx = eval_on_true_concepts(fx, vl, spec, device)   # Boolean ceiling
        tt_gl = eval_on_true_concepts(gl, vl, spec, device)
        tt_fw = eval_on_true_concepts(fw, vl, spec, device)

        print(f"{'model':28s} {'task(pred)':>11s} {'task(TRUE cpt)':>14s} {'concept':>9s}")
        print(f"{'BaconCBM (fixed/boolean)':28s} {t_fx*100:10.2f} {tt_fx*100:13.2f} {c_fx*100:8.2f}")
        print(f"{'BaconCBM (ft, gl.generic)':28s} {t_gl*100:10.2f} {tt_gl*100:13.2f} {c_gl*100:8.2f}")
        print(f"{'BaconCBM (ft, full_weight)':28s} {t_fw*100:10.2f} {tt_fw*100:13.2f} {c_fw*100:8.2f}")

        for name, tp, tt in [("gl.generic", t_gl, tt_gl), ("full_weight", t_fw, tt_fw)]:
            gap = (tp - tt) * 100
            tag = ("LEAKAGE" if gap > 3.0 else "genuine GL expressiveness (no leak)")
            print(f"  {name:12s}: pred {tp*100:.1f} vs TRUE {tt*100:.1f}  ->  {gap:+.1f} pt  [{tag}]")

        # show what fine-tuning did to a couple of trees (full_weight backend)
        print("\nfull_weight fine-tuned tree parameters (init: AND a=0.85, OR a=0.15):")
        desc = fw.logic.describe()
        for k in (0, 2, 7):
            if k in desc:
                nodes = desc[k]
                summary = "; ".join(
                    f"{n['op']} a={n['andness']:.2f} w={n['weights']}" for n in nodes)
                print(f"  class {k} ({fc.CLASS_NAMES[k]}): {spec.formulas[k]}")
                print(f"      -> {summary}")


if __name__ == "__main__":
    main()
