"""
Review the BACON trees, check concept completeness, and verify save / reuse.

    python review_trees.py

For each setting (iFMNIST, sFMNIST, cFMNIST) this prints the per-class BACON
tree, reports whether the concept set is COMPLETE (every class has a distinct
concept vector) or INCOMPLETE (some classes collide), and then round-trips a
BaconCBM through save -> load to confirm the trees + weights are reusable.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import fmnist_concepts as fc                     # noqa: E402
from models import BaconCBM                       # noqa: E402


def review(spec):
    print(f"\n{'='*72}\n{spec.name}: {spec.K} concepts "
          f"({len(spec.mutex_groups)} mutex groups, "
          f"{len(spec.binary_concepts)} binary attrs)\n{'='*72}")
    print("concepts:", ", ".join(spec.concept_names))
    print("\nper-class BACON trees:")
    for c in range(10):
        print(f"  {fc.CLASS_NAMES[c]:11s} = {spec.formulas[c]}")

    # Completeness: distinct ground-truth concept vectors across classes.
    seen, collisions = {}, []
    for c in range(10):
        key = tuple(spec.Y[c].tolist())
        if key in seen:
            collisions.append((seen[key], c))
        else:
            seen[key] = c
    n_distinct = len(seen)
    if collisions:
        pretty = ", ".join(f"{fc.CLASS_NAMES[a]}~{fc.CLASS_NAMES[b]}"
                           for a, b in collisions)
        print(f"\n  -> INCOMPLETE: {n_distinct}/10 distinct concept vectors "
              f"(collisions: {pretty})")
    else:
        print(f"\n  -> COMPLETE: all 10 classes have distinct concept vectors")
    return n_distinct


def roundtrip(spec):
    """Save an (untrained) BaconCBM and reload it; outputs must match exactly."""
    torch.manual_seed(0)
    model = BaconCBM(spec, d_y=0)
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out1, _ = model(x)
    path = os.path.join(tempfile.gettempdir(), f"bacon_{spec.name}.pt")
    model.save(path)
    reloaded, spec2 = BaconCBM.load(path, device="cpu")
    reloaded.eval()
    with torch.no_grad():
        out2, _ = reloaded(x)
    ok = torch.allclose(out1, out2, atol=1e-6)
    same_trees = spec2.formulas == spec.formulas
    print(f"  save/reuse round-trip: outputs_match={ok}  trees_preserved={same_trees}")
    os.remove(path)
    return ok and same_trees


def main():
    for spec in (fc.IFMNIST, fc.SFMNIST, fc.CFMNIST):
        review(spec)
        roundtrip(spec)

    # If a trained model was saved by run_cream, show it can be reused.
    saved = os.path.join(_HERE, "saved", "bacon_sFMNIST.pt")
    if os.path.exists(saved):
        model, spec = BaconCBM.load(saved, device="cpu")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nReused trained model {saved}: {spec.name}, "
              f"{n_params} params, {len(spec.formulas)} trees restored.")


if __name__ == "__main__":
    main()
