"""
Generalizable concept characterization (domain-agnostic-ish).

Given a saved emergent model, characterize each unnamed concept WITHOUT
hand-designing a bespoke probe, by two mechanical steps:

  1. EXEMPLARS (discovery, no priors): rank the real test set by each concept
     and report which CLASSES maximize / minimize it (+ save a top/bottom image
     grid).  "What real inputs light this concept up?"

  2. BLIND FACTOR SWEEP (causal validation): run a reusable LIBRARY of parametric
     interventions (closure, curvature, size, sharpness, complexity) and, for
     each concept, report the factor it tracks most MONOTONICALLY (Spearman rho).
     The concept "claims" whichever factor it follows -- no per-domain code.

This is the automatable core of the hypothesize->ground->causally-test loop:
exemplars propose, the monotonic sweep confirms.  Add a labelled attribute set
(CUB/CelebA) and step 1 becomes an attribute-AUC auto-namer (see
eval_concept_identification.py).

    python characterize_concepts.py --load saved/k3_trainable.pt
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys

import torch
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from shapes import _SS, _MNIST_MEAN, _MNIST_STD                 # noqa: E402
from train import make_loaders                                  # noqa: E402
from train_emergent_concepts import MultiTreeBaconCBM           # noqa: E402
from probe_c1_sweep import arc_img, bend_img, _to_tensor, _CANVAS  # noqa: E402


# --------------------------------------------------------------------------- #
# reusable FACTOR LIBRARY: name -> (generator(param, rng), params, direction)
# --------------------------------------------------------------------------- #
def size_img(rfrac, rng):
    R = rfrac * (_CANVAS * 0.42)
    cx = cy = _CANVAS / 2
    w = rng.randint(2, 3) * _SS
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    ImageDraw.Draw(img).ellipse([cx - R, cy - R, cx + R, cy + R], outline=255, width=w)
    return _to_tensor(img)


def sharp_img(sharpness, rng):
    """A corner whose interior angle goes 180deg (straight) -> 30deg (sharp)."""
    interior = math.radians(180 - sharpness * 150)
    half = interior / 2
    L = 9 * _SS
    cx, cy = _CANVAS / 2, _CANVAS / 2
    base = rng.uniform(0, 2 * math.pi)

    def rot(px, py):
        dx, dy = px - cx, py - cy
        return (cx + dx * math.cos(base) - dy * math.sin(base),
                cy + dx * math.sin(base) + dy * math.cos(base))
    v = (cx, cy - L * 0.3)
    p1 = rot(cx - L * math.sin(half), cy - L * 0.3 + L * math.cos(half))
    p2 = rot(cx + L * math.sin(half), cy - L * 0.3 + L * math.cos(half))
    v = rot(*v)
    w = rng.randint(2, 3) * _SS
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    ImageDraw.Draw(img).line([p1, v, p2], fill=255, width=w, joint="curve")
    return _to_tensor(img)


def complexity_img(n_seg, rng):
    """Zigzag with n_seg sharp segments (more segments = more complex)."""
    theta = math.radians(115)
    L = 5.0 * _SS
    ang = rng.uniform(0, 2 * math.pi)
    x = y = 0.0
    pts = [(0.0, 0.0)]
    for i in range(int(n_seg)):
        x += L * math.cos(ang); y += L * math.sin(ang)
        pts.append((x, y))
        ang += theta if i % 2 == 0 else -theta
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    s = (0.7 * _CANVAS) / max(bw, bh, 1e-6)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    pts = [((px - cx) * s + _CANVAS / 2, (py - cy) * s + _CANVAS / 2) for px, py in pts]
    w = rng.randint(2, 3) * _SS
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    ImageDraw.Draw(img).line(pts, fill=255, width=w, joint="curve")
    return _to_tensor(img)


FACTORS = {
    "closure":    (arc_img,        [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]),
    "curvature":  (bend_img,       [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    "size":       (size_img,       [0.45, 0.55, 0.65, 0.8, 0.95]),
    "sharpness":  (sharp_img,      [0.0, 0.25, 0.5, 0.75, 1.0]),
    "complexity": (complexity_img, [1, 2, 3, 4, 5]),
}


def spearman(xs, ys):
    def rank(v):
        t = torch.tensor(v, dtype=torch.float)
        return t.argsort().argsort().float()
    rx, ry = rank(xs), rank(ys)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return (rx * ry).sum().item() / ((rx.norm() * ry.norm()).item() + 1e-8)


def onset(params, ys):
    """Normalized (0..1) param at which the response first crosses its own
    midpoint -- the 'timing' of a monotonic response.  Small = early riser
    (responds to the factor's onset), large = late riser (needs the factor near
    its extreme).  This disambiguates concepts that share |rho|~1 on correlated
    factors (e.g. curvature rises EARLY, closedness rises LATE)."""
    y = torch.tensor(ys, dtype=torch.float)
    lo, hi = y.min(), y.max()
    if (hi - lo).item() < 0.10:                       # essentially flat -> no onset
        return float("nan")
    mid = (lo + hi) / 2
    p0, p1 = params[0], params[-1]
    for k in range(1, len(y)):
        if (y[k - 1] - mid) * (y[k] - mid) <= 0:
            frac = ((mid - y[k - 1]) / (y[k] - y[k - 1] + 1e-8)).item()
            pv = params[k - 1] + frac * (params[k] - params[k - 1])
            return (pv - p0) / (p1 - p0 + 1e-8)
    return 1.0 if y[-1] > mid else 0.0


# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--n", type=int, default=200, help="samples per sweep point")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", type=str, default=os.path.join(_HERE, "results"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.load, map_location=device, weights_only=False)
    K = ckpt["K"]
    model = MultiTreeBaconCBM(K, weight_mode=ckpt.get("weight_mode", "trainable"),
                              device=device).to(device)
    if ckpt.get("frozen"):
        model.prepare_frozen_structure(); model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt["state_dict"]); model.anneal(1.0)
    model.eval()
    print(f"loaded {args.load}  K={K}  saved-acc={ckpt.get('acc', float('nan')) * 100:.2f}%")

    _, test_ld = make_loaders(os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data"), 512)
    imgs, ys, cs = [], [], []
    for x, y in test_ld:
        imgs.append(x); ys.append(y)
        cs.append(model.concept_probs(x.to(device)).cpu())
    X = torch.cat(imgs); Y = torch.cat(ys); C = torch.cat(cs)          # (N,1,28,28),(N,),(N,K)

    # ---- 1. EXEMPLARS ------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("EXEMPLARS  (which digit classes max / min each concept)")
    print("=" * 70)
    grid = Image.new("L", (2 * args.topk * 28, K * 28), 0)
    for i in range(K):
        order = C[:, i].argsort(descending=True)
        top, bot = order[:args.topk], order[-args.topk:]

        def dist(idx):
            c = collections.Counter(Y[idx].tolist())
            return " ".join(f"{d}:{n}" for d, n in sorted(c.items(), key=lambda kv: -kv[1]))
        print(f"  c{i}:  TOP  [{dist(top)}]")
        print(f"        BOT  [{dist(bot)}]")
        for j, idx in enumerate(list(top) + list(bot)):
            im = (X[idx, 0] * _MNIST_STD + _MNIST_MEAN).clamp(0, 1).mul(255).byte().numpy()
            grid.paste(Image.fromarray(im, "L"), (j * 28, i * 28))
    p = os.path.join(args.out, "exemplars.png")
    grid.save(p)
    print(f"  (saved top/bottom-{args.topk} grid -> {p})")

    # ---- 2. BLIND FACTOR SWEEP -------------------------------------------- #
    print("\n" + "=" * 70)
    print("BLIND FACTOR SWEEP  (Spearman rho + onset-timing to disambiguate)")
    print("=" * 70)
    curves = {}
    for fname, (fn, params) in FACTORS.items():
        resp = []
        for pval in params:
            rng = random.Random(123)
            b = torch.stack([fn(pval, rng) for _ in range(args.n)], 0).to(device)
            resp.append(model.concept_probs(b).mean(0).cpu())     # (K,)
        curves[fname] = (params, torch.stack(resp))               # (P,K)

    # rho table
    print("\n  rho (monotonicity, |rho|~1 = tracks factor):")
    print("  concept  " + "".join(f"{f:>12s}" for f in FACTORS))
    rhos = {}
    for i in range(K):
        row = []
        for fname in FACTORS:
            params, resp = curves[fname]
            r = spearman(params, resp[:, i].tolist())
            rhos[(i, fname)] = r
            row.append(f"{r:+.2f}")
        print(f"  c{i}     " + "".join(f"{c:>12s}" for c in row))

    # onset table (only where the concept actually moves on that factor)
    print("\n  onset (0=early riser/direct, 1=late riser; '--' = flat):")
    print("  concept  " + "".join(f"{f:>12s}" for f in FACTORS))
    onsets = {}
    for i in range(K):
        row = []
        for fname in FACTORS:
            params, resp = curves[fname]
            o = onset(params, resp[:, i].tolist())
            onsets[(i, fname)] = o
            row.append("--" if o != o else f"{o:.2f}")
        print(f"  c{i}     " + "".join(f"{c:>12s}" for c in row))

    # disambiguated characterization: among factors the concept tracks (|rho|>=0.9,
    # non-flat), prefer the one it responds to EARLIEST (smallest onset) = the most
    # direct/primitive factor, breaking the closure/curvature/size confound ties.
    print("\n  => characterization (best |rho|, ties broken by earliest onset):")
    for i in range(K):
        cand = [(f, rhos[(i, f)], onsets[(i, f)]) for f in FACTORS
                if abs(rhos[(i, f)]) >= 0.9 and onsets[(i, f)] == onsets[(i, f)]]
        if not cand:
            f, r = max(FACTORS, key=lambda f: abs(rhos[(i, f)])), None
            print(f"     c{i}: ~ {f} (rho {rhos[(i, f)]:+.2f}, weak/flat)")
            continue
        f, r, o = min(cand, key=lambda t: t[2])           # earliest onset among strong
        others = ", ".join(f"{g}@{onsets[(i, g)]:.2f}" for g, rr, oo in
                           sorted(cand, key=lambda t: t[2]) if g != f)
        print(f"     c{i}: {f} (rho {r:+.2f}, onset {o:.2f})"
              + (f"   [also tracks {others}]" if others else ""))
    print("\n  note: closure/curvature/size co-vary in shape space; onset (timing)")
    print("        separates the DIRECT factor (early) from indirect ones (late).")


if __name__ == "__main__":
    main()
