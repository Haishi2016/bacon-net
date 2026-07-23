"""
Cross-model zero-shot concept transfer: run the SAME blob->blob fashion zero-shot
across all five FashionMNIST models.

Trains (once, cached) BlackBox, SoftCBM, CREAM, BaconCBM and BaconCBM+SC on the
FashionMNIST sFMNIST setting, then applies each to clothing-dataset-small product
photos rendered as FashionMNIST-style *shaded* silhouettes (see
eval_fashion_photos.py) with NO retraining.  Every model outputs a 10-way
FashionMNIST class; we map the predicted class to its sub-category concept and
check it against the expected sub-category for each clothing class.

    python eval_zeroshot_models.py --seeds 0,1,2 --epochs 30

Reports, averaged over seeds, each model's in-distribution FashionMNIST test
accuracy and its zero-shot sub-category transfer (mean +/- std matches / 10),
plus a per-class hit-rate (fraction of seeds correct) to show robustness.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import fmnist_concepts as fc                       # noqa: E402
import run_cream as rc                             # noqa: E402
import eval_fashion_photos as efp                  # noqa: E402
from models import (BlackBox, SoftCBM, CREAM, BaconCBM,   # noqa: E402
                    save_checkpoint, load_checkpoint)

SPEC = fc.SFMNIST


def _configs():
    """(name, builder, has_concepts, save-kwargs)."""
    return [
        ("BlackBox",    lambda: BlackBox(),                         False, dict(d_y=0)),
        ("SoftCBM",     lambda: SoftCBM(SPEC),                      True,  dict(d_y=0)),
        ("CREAM",       lambda: CREAM(SPEC, dropout_p=0.9),         True,  dict(d_c=7, d_y=20, dropout_p=0.9)),
        ("BaconCBM",    lambda: BaconCBM(SPEC),                     True,  dict(d_y=0)),
        ("BaconCBM+SC", lambda: BaconCBM(SPEC, d_y=20, dropout_p=0.9), True, dict(d_y=20, dropout_p=0.9)),
    ]


def ensure_model(name, builder, has_c, kw, seed, save_dir, tl, vl, device, epochs):
    """Train (or load cached) one model for a given seed."""
    path = os.path.join(save_dir, f"zs_{name}_s{seed}.pt")
    if os.path.exists(path):
        model, _, _, acc = load_checkpoint(path, device)
        return model, acc
    torch.manual_seed(seed)
    model = builder().to(device)
    rc.train_model(model, tl, SPEC, device, epochs, 1.0, has_c)
    acc, _, _ = rc.evaluate(model, vl, SPEC, device, name != "BlackBox")
    spec = None if name == "BlackBox" else SPEC
    save_checkpoint(model, path, name, spec=spec, fmnist_acc=acc, **kw)
    return model, acc


def subcat_map(spec):
    g2 = {spec.concept_names[i] for i in spec.mutex_groups[1]}
    m = {}
    for cls, on in spec.class_on.items():
        for n in on:
            if n in g2:
                m[cls] = n
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--data", type=str,
                    default=os.path.join(os.path.dirname(_HERE), "..", "..",
                                         "benchmarks", "mnist-addition", "data"))
    ap.add_argument("--save-dir", type=str, default=os.path.join(_HERE, "saved"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    tl, vl = rc.load_fmnist(args.data, 256)

    base = efp.ensure_dataset(os.path.join(_HERE, "cache_clothing"))
    cls_sub = subcat_map(SPEC)
    cats = list(efp.CATEGORIES.items())

    # Pre-render the shaded silhouettes once (shared across models + seeds).
    silhouettes = {}
    for cat, _ in cats:
        try:
            silhouettes[cat] = efp.load_class(base, cat, args.n, "test", shaded=True).to(device)
        except FileNotFoundError:
            pass
    cats = [(c, e) for c, e in cats if c in silhouettes]
    total = len(cats)

    names = [c[0] for c in _configs()]
    acc_by = {n: [] for n in names}            # FMNIST test acc per seed
    match_by = {n: [] for n in names}          # zero-shot matches (/total) per seed
    hit_by = {n: {c: 0 for c, _ in cats} for n in names}   # per-class hits over seeds

    for seed in seeds:
        print(f"--- seed {seed} ---")
        for name, builder, has_c, kw in _configs():
            model, acc = ensure_model(name, builder, has_c, kw, seed,
                                      args.save_dir, tl, vl, device, args.epochs)
            model.eval()
            acc_by[name].append(acc)
            m = 0
            with torch.no_grad():
                for cat, expected in cats:
                    out = model(silhouettes[cat])
                    logits = out[0] if isinstance(out, tuple) else out
                    idx = logits.argmax(1).tolist()
                    sub = Counter(cls_sub[i] for i in idx).most_common(1)[0][0]
                    hit = sub == expected
                    m += int(hit)
                    hit_by[name][cat] += int(hit)
            match_by[name].append(m)

    ns = len(seeds)

    def ms(v):
        t = torch.tensor(v, dtype=torch.float)
        return t.mean().item(), (t.std(unbiased=False).item() if len(v) > 1 else 0.0)

    print(f"\n{'='*70}\nAveraged over {ns} seeds {seeds}  (FashionMNIST sFMNIST, "
          f"{args.epochs} ep)\n{'='*70}")
    print(f"\n{'model':12s} {'FMNIST test acc':>18s} {'zero-shot matches /'+str(total):>22s}")
    order = sorted(names, key=lambda n: -ms(match_by[n])[0])
    for n in order:
        am, asd = ms(acc_by[n]); mm, msd = ms(match_by[n])
        print(f"{n:12s} {am*100:8.2f} +/- {asd*100:4.2f}    "
              f"{mm:6.2f} +/- {msd:4.2f}")

    print(f"\nPer-class hit-rate over {ns} seeds (fraction correct):")
    header = f"{'class':11s} {'expected':11s} " + " ".join(f"{n[:11]:>12s}" for n in names)
    print(header)
    for cat, expected in cats:
        row = f"{cat:11s} {expected:11s} "
        for n in names:
            row += f" {hit_by[n][cat]/ns:11.2f}"
        print(row)


if __name__ == "__main__":
    main()
