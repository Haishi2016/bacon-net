"""
Run the frozen "0" (circle) detector on Google Quick, Draw! everyday objects.

Zero-shot: the MNIST-trained concept encoder + the frozen "0" BACON tree
(loop_upper AND loop_lower AND NOT horizontal_middle AND NOT vertical_line) are
applied directly to doodles of real everyday objects.  We compare naturally
ROUND objects (clock, wheel, donut, ...) against NON-ROUND ones (ladder,
envelope, pants, ...) and read the "0" tree truth as a circle score.

    python eval_quickdraw.py            # downloads a few hundred imgs/category

Nothing is trained on Quick, Draw!.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

import quickdraw as qd                          # noqa: E402
from eval_shapes import load_model, roc_auc, best_threshold_accuracy  # noqa: E402

ROUND = ["circle", "donut", "clock", "wheel", "cookie", "basketball", "pizza"]
NON_ROUND = ["ladder", "envelope", "line", "pants", "table", "fork", "pencil", "zigzag"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(_HERE, "checkpoint.pt"))
    ap.add_argument("--n", type=int, default=500, help="images per category")
    ap.add_argument("--round", nargs="*", default=ROUND)
    ap.add_argument("--non-round", nargs="*", default=NON_ROUND)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}\nrun train.py first.")
    model = load_model(args.ckpt, device)
    ci = model.logic.concept_index
    lu, ll, vl, hm = (ci["loop_upper"], ci["loop_lower"],
                      ci["vertical_line"], ci["horizontal_middle"])

    cats = [(c, 1) for c in args.round] + [(c, 0) for c in args.non_round]
    preview_imgs = []
    rows = []
    all_scores, all_labels = [], []

    print(f"\ncheckpoint: {args.ckpt}   {args.n} imgs/category   device: {device}")
    print("'0'-tree = loop_upper AND loop_lower AND NOT horizontal_middle "
          "AND NOT vertical_line\n")
    print(f"{'category':12s} {'round?':>6s} {'0-score':>8s} "
          f"{'loopU':>6s} {'loopL':>6s} {'vert':>6s} {'midBar':>6s}")

    for cat, is_round in cats:
        try:
            imgs = qd.load_category(cat, n=args.n).to(device)
        except Exception as e:
            print(f"{cat:12s}  skip ({e})")
            continue
        with torch.no_grad():
            _, probs, truths = model(imgs)
        s = truths[:, 0]
        all_scores.append(s.cpu())
        all_labels.append(torch.full((len(s),), is_round))
        rows.append((cat, is_round, s.mean().item()))
        print(f"{cat:12s} {is_round:6d} {s.mean():8.3f} "
              f"{probs[:, lu].mean():6.2f} {probs[:, ll].mean():6.2f} "
              f"{probs[:, vl].mean():6.2f} {probs[:, hm].mean():6.2f}")
        if args.preview:
            preview_imgs.append((cat, imgs[:8].cpu()))

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    auc = roc_auc(scores, labels)
    acc, thr = best_threshold_accuracy(scores, labels)
    print(f"\nround vs non-round  --  ROC-AUC {auc:.3f} | "
          f"best acc {acc:.3f} @ thr {thr:.3f}")

    rows.sort(key=lambda r: -r[2])
    print("\nranked by '0'-score (circle-likeness):")
    for cat, is_round, sc in rows:
        tag = "round" if is_round else "     "
        print(f"  {sc:.3f}  {tag}  {cat}")

    if args.preview and preview_imgs:
        from PIL import Image
        cols = 8
        grid = Image.new("L", (cols * 28, len(preview_imgs) * 28), 0)
        for r, (cat, imgs) in enumerate(preview_imgs):
            for c in range(min(cols, len(imgs))):
                raw = (imgs[c] * qd._MNIST_STD + qd._MNIST_MEAN).clamp(0, 1)
                arr = raw.mul(255).byte().squeeze().numpy()
                grid.paste(Image.fromarray(arr, "L"), (c * 28, r * 28))
        p = os.path.join(_HERE, "quickdraw_preview.png")
        grid.save(p)
        print(f"\nsaved preview (one row per category) -> {p}")


if __name__ == "__main__":
    main()
