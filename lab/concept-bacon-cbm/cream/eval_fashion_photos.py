"""
Blob-with-blob zero-shot concept validation for the FASHION model.

Line-based models don't generalise to blob-based data (and vice-versa), so we
validate the FashionMNIST BACON model on ANOTHER blob/silhouette clothing source
rather than on line doodles.  We use `clothing-dataset-small` (10 clothing
classes: t-shirt, shirt, longsleeve, outwear, dress, pants, shorts, skirt,
shoes, hat), convert each product photo into a FashionMNIST-style filled
silhouette (grayscale -> border-background removal -> largest component -> fill
-> bbox-normalise -> FashionMNIST norm), and apply the saved model with NO
retraining.

    python eval_fashion_photos.py --preview

Reports the predicted FashionMNIST class + fired concepts per clothing class and
whether the sub-category concept matches the expected one.

RESULT (honest): keeping the garment's grayscale interior (shaded, default)
roughly DOUBLES transfer over a flat binary mask: **6/10 vs 3/10** sub-categories,
with several correct at the class level (t-shirt->T-shirt, shirt->Shirt,
dress->Dress, pants->Trouser, outwear->Coat) and the long_sleeve attribute firing
on shirt/longsleeve.  This confirms the domain point: blob-with-blob transfers
well *when the appearance matches* -- FashionMNIST relies on internal shading, so
a flat silhouette underperforms while a shaded one bridges the gap.  Remaining
misses (shorts, skirt, shoes) are the hardest silhouettes.  Run `--binary` to
reproduce the flat-mask baseline.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import fmnist_concepts as fc                       # noqa: E402
from models import BaconCBM                         # noqa: E402

_FM_MEAN, _FM_STD = 0.2860, 0.3530
_URL = "https://github.com/alexeygrigorev/clothing-dataset-small/archive/refs/heads/master.zip"

# clothing-dataset-small class -> expected FashionMNIST sub-category concept.
CATEGORIES = {
    "t-shirt": "Tops", "shirt": "Tops", "longsleeve": "Tops",
    "outwear": "Outers", "dress": "Dresses",
    "pants": "Bottoms", "shorts": "Bottoms", "skirt": "Bottoms",
    "shoes": "Shoes", "hat": "Accessories",
}


def ensure_dataset(cache: str) -> str:
    os.makedirs(cache, exist_ok=True)
    base = os.path.join(cache, "clothing-dataset-small-master")
    if os.path.isdir(base):
        return base
    z = os.path.join(cache, "clothing.zip")
    if not os.path.exists(z):
        print("downloading clothing-dataset-small (~40 MB)...")
        urllib.request.urlretrieve(_URL, z)
    zipfile.ZipFile(z).extractall(cache)
    return base


def photo_to_silhouette(path: str, canvas: int = 96, fill_size: int = 24,
                        shaded: bool = True) -> torch.Tensor:
    """A clothing product photo -> FashionMNIST-style silhouette (1,28,28).

    shaded=True keeps the garment's grayscale interior (internal shading/texture,
    oriented light-on-dark like FashionMNIST); shaded=False is a flat binary mask.
    """
    im = Image.open(path).convert("L")
    im.thumbnail((canvas, canvas), Image.BILINEAR)          # keep aspect
    g = np.asarray(im, dtype=np.float32) / 255.0
    # paste centered on a background-coloured square (bg = border median)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = float(np.median(border))
    sq = np.full((canvas, canvas), bg, dtype=np.float32)
    h, w = g.shape
    y0, x0 = (canvas - h) // 2, (canvas - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = g

    fg = np.abs(sq - bg) > 0.15                             # foreground vs background
    fg = ndimage.binary_closing(fg, iterations=2)
    fg = ndimage.binary_fill_holes(fg)
    lbl, n = ndimage.label(fg)                              # keep largest component
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        fg = lbl == (1 + int(np.argmax(sizes)))

    if shaded:
        # garment deviation from background = light-on-dark shading, bg -> 0
        shape = np.abs(sq - bg) * fg
        hi = np.percentile(shape[fg], 98) if fg.any() else 1.0
        shape = np.clip(shape / (hi + 1e-6), 0.0, 1.0).astype(np.float32)
    else:
        shape = fg.astype(np.float32)

    ys, xs = np.where(fg)
    out = torch.zeros(1, 28, 28)
    if len(xs):
        crop = shape[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ch, cw = crop.shape
        s = fill_size / max(ch, cw)
        nh, nw = max(1, round(ch * s)), max(1, round(cw * s))
        r = F.interpolate(torch.from_numpy(crop)[None, None], size=(nh, nw),
                          mode="bilinear", align_corners=False)[0, 0]
        yy, xx = (28 - nh) // 2, (28 - nw) // 2
        out[0, yy:yy + nh, xx:xx + nw] = r
    return (out - _FM_MEAN) / _FM_STD


def load_class(base: str, cat: str, n: int, split: str = "test",
               shaded: bool = True) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(base, split, cat, "*.jpg")))[:n]
    if not paths:
        raise FileNotFoundError(cat)
    return torch.stack([photo_to_silhouette(p, shaded=shaded) for p in paths], 0)


def _mode(names):
    from collections import Counter
    return Counter(names).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(_HERE, "saved", "bacon_sFMNIST.pt"))
    ap.add_argument("--cache", type=str, default=os.path.join(_HERE, "cache_clothing"))
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--binary", action="store_true",
                    help="flat binary mask instead of shaded grayscale interior")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"missing {args.ckpt}; run run_cream.py first.")
    model, spec = BaconCBM.load(args.ckpt, device=device)
    model.eval()
    g1, g2 = spec.mutex_groups[0], spec.mutex_groups[1]
    binaries = spec.binary_concepts
    # map each FashionMNIST class -> its sub-category concept (group-2 membership)
    g2_names = {spec.concept_names[i] for i in g2}
    cls_sub = {}
    for cls, on in spec.class_on.items():
        for name in on:
            if name in g2_names:
                cls_sub[cls] = name
    base = ensure_dataset(args.cache)

    print(f"\nFashionMNIST BACON model -> clothing-dataset-small (blob->blob, zero-shot)")
    print(f"{'class':11s} {'pred':11s} {'cat':7s} {'sub-cat':11s} {'match':5s}  attrs")
    correct = total = 0
    preview = []
    for cat, expected in CATEGORIES.items():
        try:
            x = load_class(base, cat, args.n, args.split, shaded=not args.binary).to(device)
        except FileNotFoundError:
            print(f"{cat:11s}  skip"); continue
        with torch.no_grad():
            logits, c = model(x)
        pred_idx = logits.argmax(1).tolist()
        pred = _mode([fc.CLASS_NAMES[p] for p in pred_idx])
        sub = _mode([cls_sub[p] for p in pred_idx])          # sub-cat of predicted class
        cp = c.mean(0)
        cat1 = spec.concept_names[g1[cp[g1].argmax()]]
        attrs = [spec.concept_names[k] for k in binaries if cp[k] > 0.5]
        ok = sub == expected
        correct += int(ok); total += 1
        print(f"{cat:11s} {pred:11s} {cat1:7s} {sub:11s} {'yes' if ok else 'no':5s}  "
              f"{', '.join(attrs) if attrs else '-'}")
        if args.preview:
            preview.append((cat, x[:8].cpu()))
    print(f"\nsub-category concept matches expected: {correct}/{total}")

    if args.preview and preview:
        grid = Image.new("L", (8 * 28, len(preview) * 28), 0)
        for r, (cat, xs) in enumerate(preview):
            for cc in range(min(8, len(xs))):
                arr = ((xs[cc] * _FM_STD + _FM_MEAN).clamp(0, 1).mul(255)
                       .byte().squeeze().numpy())
                grid.paste(Image.fromarray(arr, "L"), (cc * 28, r * 28))
        p = os.path.join(_HERE, "fashion_photo_silhouettes.png")
        grid.save(p)
        print(f"saved preview -> {p}")


if __name__ == "__main__":
    main()
