"""
Run the frozen "0" (circle) detector on REAL object photos.

The MNIST-trained concept encoder only understands 28x28 white-stroke-on-black
images, so a raw color photo is far out of distribution.  The bridge is an edge
transform (see photo_sketch.py): a photo of a round object becomes a white
circular outline on black -- back in the "hand-drawn 0" domain.

Two well-known real-photo datasets are supported (both fetched from the fast
fast.ai S3 mirror, so no slow Toronto download):

  --dataset caltech101   (default) object-centric photos, higher resolution.
        round     = soccer_ball, watch, yin_yang, stop_sign, pizza, sunflower
        non-round = laptop, scissors, chair, electric_guitar, wrench, airplanes
  --dataset cifar100     32x32 cluttered photos (harder; near-chance).
        round     = plate, clock, apple, orange, bowl, sunflower
        non-round = keyboard, table, bridge, train, rocket, wardrobe

    python eval_photos.py --preview

Nothing is trained on the photos.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tarfile
import urllib.request

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

import photo_sketch                              # noqa: E402
from bacon_logic import And, Concept             # noqa: E402
from eval_shapes import load_model, roc_auc, best_threshold_accuracy  # noqa: E402

_BASE = "https://s3.amazonaws.com/fast-ai-imageclas/{}.tgz"

DATASETS = {
    "caltech101": {
        "file": "caltech_101",
        "glob": os.path.join("caltech101_ex", "101_ObjectCategories", "{cat}", "*.jpg"),
        "round": ["soccer_ball", "watch", "yin_yang", "stop_sign", "pizza", "sunflower"],
        "non_round": ["laptop", "scissors", "chair", "electric_guitar", "wrench", "airplanes"],
        "resize": 96,
    },
    "cifar100": {
        "file": "cifar100",
        "glob": os.path.join("cifar100_ex", "cifar100", "test", "*", "{cat}", "*.png"),
        "round": ["plate", "clock", "apple", "orange", "bowl", "sunflower"],
        "non_round": ["keyboard", "table", "bridge", "train", "rocket", "wardrobe"],
        "resize": 32,
    },
}


def ensure_dataset(spec: dict, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    ex_dir = os.path.join(cache_dir, spec["glob"].split(os.sep)[0])
    if os.path.isdir(ex_dir):
        return
    tgz = os.path.join(cache_dir, spec["file"] + ".tgz")
    if not os.path.exists(tgz):
        print(f"downloading {spec['file']} (fast.ai mirror)...")
        urllib.request.urlretrieve(_BASE.format(spec["file"]), tgz)
    print("extracting...")
    with tarfile.open(tgz) as tf:
        tf.extractall(ex_dir)


def load_class(spec: dict, cache_dir: str, cat: str, n: int) -> torch.Tensor:
    """Load up to n real photos of a class, center-square, resized -> (m,3,R,R)."""
    pattern = os.path.join(cache_dir, spec["glob"].format(cat=cat))
    paths = sorted(glob.glob(pattern))[:n]
    if not paths:
        raise FileNotFoundError(f"no images for class '{cat}'")
    R = spec["resize"]
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = min(w, h)                                    # center square crop
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            im = im.resize((R, R), Image.BILINEAR)
            t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            t = t.float().reshape(R, R, 3).permute(2, 0, 1) / 255.0
        imgs.append(t)
    return torch.stack(imgs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="caltech101")
    ap.add_argument("--ckpt", type=str, default=os.path.join(_HERE, "checkpoint.pt"))
    ap.add_argument("--cache", type=str, default=os.path.join(_HERE, "cache"))
    ap.add_argument("--n", type=int, default=100, help="images per class")
    ap.add_argument("--round", nargs="*", default=None)
    ap.add_argument("--non-round", nargs="*", default=None)
    ap.add_argument("--keep-frac", type=float, default=0.12,
                    help="target fraction of stroke pixels (MNIST ~0.13)")
    ap.add_argument("--blur", type=int, default=2, help="Gaussian blur passes")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    round_cats = args.round or spec["round"]
    non_round_cats = args.non_round or spec["non_round"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}\nrun train.py first.")
    model = load_model(args.ckpt, device)
    ci = model.logic.concept_index
    lu, ll, vl, hm = (ci["loop_upper"], ci["loop_lower"],
                      ci["vertical_line"], ci["horizontal_middle"])

    ensure_dataset(spec, args.cache)

    # "loop-only" roundness score = the loop_upper AND loop_lower part of the
    # "0" rule, WITHOUT the empty-interior (NOT vert / NOT midBar) clauses.
    loop_node = And([Concept("loop_upper"), Concept("loop_lower")])

    cats = [(c, 1) for c in round_cats] + [(c, 0) for c in non_round_cats]
    all_scores, all_labels, all_loop, rows, preview = [], [], [], [], []

    print(f"\ndataset: {args.dataset}   {args.n} imgs/class   device: {device}")
    print("real photos -> Sobel edge sketch -> '0' tree "
          "(loopU AND loopL AND NOT midBar AND NOT vert)\n")
    print(f"{'class':15s} {'round?':>6s} {'0-score':>8s} {'loopAND':>8s} "
          f"{'loopU':>6s} {'loopL':>6s} {'vert':>6s} {'midBar':>6s}")

    for cat, is_round in cats:
        try:
            imgs = load_class(spec, args.cache, cat, args.n).to(device)
        except FileNotFoundError as e:
            print(f"{cat:15s}  skip ({e})")
            continue
        sketch = photo_sketch.to_sketch(imgs, blur=args.blur, keep_frac=args.keep_frac)
        with torch.no_grad():
            _, probs, truths = model(sketch)
            loop = model.logic._eval(loop_node, probs)
        s = truths[:, 0]
        all_scores.append(s.cpu())
        all_loop.append(loop.cpu())
        all_labels.append(torch.full((len(s),), is_round))
        rows.append((cat, is_round, s.mean().item(), loop.mean().item()))
        print(f"{cat:15s} {is_round:6d} {s.mean():8.3f} {loop.mean():8.3f} "
              f"{probs[:, lu].mean():6.2f} {probs[:, ll].mean():6.2f} "
              f"{probs[:, vl].mean():6.2f} {probs[:, hm].mean():6.2f}")
        if args.preview:
            preview.append((cat, sketch[:8].cpu()))

    scores = torch.cat(all_scores)
    loops = torch.cat(all_loop)
    labels = torch.cat(all_labels)
    auc = roc_auc(scores, labels)
    acc, thr = best_threshold_accuracy(scores, labels)
    loop_auc = roc_auc(loops, labels)
    print(f"\nfull '0' tree (empty ring)   round vs non-round -- ROC-AUC {auc:.3f} "
          f"| best acc {acc:.3f} @ thr {thr:.3f}")
    print(f"loop-only (round outline)    round vs non-round -- ROC-AUC {loop_auc:.3f}")

    rows.sort(key=lambda r: -r[3])
    print("\nranked by loop-only roundness (loopU AND loopL):")
    for cat, is_round, sc, lp in rows:
        print(f"  {lp:.3f}  {'round' if is_round else '     '}  {cat}")

    if args.preview and preview:
        cols = 8
        grid = Image.new("L", (cols * 28, len(preview) * 28), 0)
        for r, (cat, sk) in enumerate(preview):
            for c in range(min(cols, len(sk))):
                raw = (sk[c] * photo_sketch._MNIST_STD + photo_sketch._MNIST_MEAN)
                arr = raw.clamp(0, 1).mul(255).byte().squeeze().numpy()
                grid.paste(Image.fromarray(arr, "L"), (c * 28, r * 28))
        p = os.path.join(_HERE, f"photos_preview_{args.dataset}.png")
        grid.save(p)
        print(f"\nsaved edge-sketch preview (one row per class) -> {p}")


if __name__ == "__main__":
    main()
