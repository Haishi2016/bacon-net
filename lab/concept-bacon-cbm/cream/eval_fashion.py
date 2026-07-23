"""
Zero-shot concept validation for the FASHION model (not digits).

We take the FashionMNIST-trained BACON model (saved/bacon_sFMNIST.pt, with the
clothing concepts Clothes/Goods, Tops/Bottoms/Dresses/Outers/Accessories/Shoes,
and long_sleeve/front_opening/open_toe/ankle_high) and apply it, with NO
retraining, to Quick, Draw! clothing doodles (t-shirt, pants, shoe, ...).

Domain bridge: FashionMNIST items are FILLED grayscale silhouettes, while
Quick, Draw! doodles are OUTLINES.  We fill each outline into a silhouette
(morphological close + fill holes) so it resembles a FashionMNIST item, then
normalise with FashionMNIST statistics.

    python eval_fashion.py --preview

We report, per doodle category, the predicted FashionMNIST class and the fired
concepts, and check the expected mapping (t-shirt -> Tops, pants -> Bottoms,
shoe -> Shoes, ...).

RESULT (honest): transfer is WEAK (~2/7 sub-categories).  Only t-shirt/Tops
reliably survives; most doodles collapse to "Bag", because a filled doodle blob
most resembles FashionMNIST's Bag class.  The appearance gap is fundamental:
FashionMNIST items are photographic, *filled and shaded* silhouettes, whereas
Quick, Draw! are sparse *line doodles* -- neither filling nor outlining bridges
it well.  A faithful fashion zero-shot target would be clothing PHOTOS processed
like FashionMNIST (background-removed, grayscale, tightly cropped), not doodles.
This mirrors the digit->CIFAR-photo domain-gap finding in ../eval_photos.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _PARENT)

import quickdraw as qd                            # noqa: E402  (parent loader)
import fmnist_concepts as fc                       # noqa: E402
from models import BaconCBM                         # noqa: E402

_FM_MEAN, _FM_STD = 0.2860, 0.3530

# Quick, Draw! clothing categories -> expected FashionMNIST sub-category concept.
CATEGORIES = {
    "t-shirt": "Tops",
    "sweater": "Tops",
    "jacket": "Outers",
    "pants": "Bottoms",
    "shorts": "Bottoms",
    "shoe": "Shoes",
    "sock": "Shoes",
}


def to_silhouette(x01: torch.Tensor, fill_size: int = 24, fill: bool = True) -> torch.Tensor:
    """(m,1,28,28) outline in [0,1] -> bbox-normalised silhouette (or outline).

    Optionally fill the outline into a solid silhouette, then crop to the
    garment's bounding box and rescale it to ~fill_size px centered in 28x28,
    matching FashionMNIST's tight framing.
    """
    out = torch.zeros(x01.shape[0], 1, 28, 28)
    for i in range(x01.shape[0]):
        b = (x01[i, 0].numpy() > 0.35)
        b = ndimage.binary_closing(b, iterations=2)
        if fill:
            shape = (ndimage.binary_fill_holes(b) | b).astype("float32")
        else:
            shape = b.astype("float32")
        ys, xs = np.where(shape > 0)
        if len(xs) == 0:
            continue
        crop = shape[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = crop.shape
        s = fill_size / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        r = F.interpolate(torch.from_numpy(crop)[None, None], size=(nh, nw),
                          mode="bilinear", align_corners=False)[0, 0]
        y0, x0 = (28 - nh) // 2, (28 - nw) // 2
        out[i, 0, y0:y0 + nh, x0:x0 + nw] = r
    return out


def load_silhouettes(cat: str, n: int, fill: bool = True) -> torch.Tensor:
    t = qd.load_category(cat, n)                            # (m,1,28,28) MNIST-normed
    x01 = (t * qd._MNIST_STD + qd._MNIST_MEAN).clamp(0, 1)  # -> [0,1] outline
    sil = to_silhouette(x01, fill=fill)
    return (sil - _FM_MEAN) / _FM_STD                       # FashionMNIST norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(_HERE, "saved", "bacon_sFMNIST.pt"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--no-fill", action="store_true",
                    help="feed bbox-normalised outlines instead of filled silhouettes")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"missing {args.ckpt}; run run_cream.py first to train+save it.")
    model, spec = BaconCBM.load(args.ckpt, device=device)
    model.eval()
    g1, g2 = spec.mutex_groups[0], spec.mutex_groups[1]     # Clothes/Goods, sub-cat
    binaries = spec.binary_concepts

    print(f"\nFashionMNIST BACON model -> Quick,Draw! clothing (zero-shot, {args.n}/cat)")
    print(f"concepts: {', '.join(spec.concept_names)}\n")
    print(f"{'doodle':9s} {'pred class':11s} {'cat':7s} {'sub-cat':11s} "
          f"{'match':5s}  fired binary attrs")

    correct = total = 0
    preview = []
    for cat, expected_sub in CATEGORIES.items():
        try:
            x = load_silhouettes(cat, args.n, fill=not args.no_fill).to(device)
        except Exception as e:
            print(f"{cat:9s}  skip ({str(e)[:40]})")
            continue
        with torch.no_grad():
            logits, c = model(x)
        pred = logits.argmax(1)
        pred_name = _mode_name([fc.CLASS_NAMES[p] for p in pred.tolist()])
        cprobs = c.mean(0)
        cat1 = spec.concept_names[g1[cprobs[g1].argmax()]]
        sub = spec.concept_names[g2[cprobs[g2].argmax()]]
        attrs = [spec.concept_names[k] for k in binaries if cprobs[k] > 0.5]
        ok = sub == expected_sub
        correct += int(ok); total += 1
        print(f"{cat:9s} {pred_name:11s} {cat1:7s} {sub:11s} "
              f"{'yes' if ok else 'no':5s}  {', '.join(attrs) if attrs else '-'}")
        if args.preview:
            preview.append((cat, x[:8].cpu()))

    print(f"\nsub-category concept matches expected: {correct}/{total}")

    if args.preview and preview:
        from PIL import Image
        grid = Image.new("L", (8 * 28, len(preview) * 28), 0)
        for r, (cat, xs) in enumerate(preview):
            for cc in range(min(8, len(xs))):
                arr = ((xs[cc] * _FM_STD + _FM_MEAN).clamp(0, 1)
                       .mul(255).byte().squeeze().numpy())
                grid.paste(Image.fromarray(arr, "L"), (cc * 28, r * 28))
        p = os.path.join(_HERE, "fashion_silhouettes.png")
        grid.save(p)
        print(f"saved silhouette preview -> {p}")


def _mode_name(names):
    from collections import Counter
    return Counter(names).most_common(1)[0][0]


if __name__ == "__main__":
    main()
