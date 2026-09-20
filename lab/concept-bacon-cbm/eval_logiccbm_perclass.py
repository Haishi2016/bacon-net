r"""Per-class test recall for the LogicCBM reproduction on CUB, evaluated in the
SAME _cub pipeline our recttree uses (apples-to-apples). Prints overall accuracy,
sanity-checks it against the checkpoint's stored acc, and reports the recall for
a requested class (default Hooded Merganser, class 88) plus the hardest classes.

  py -3 eval_logiccbm_perclass.py --ckpt saved/logiccbm_k312_250n_inception_v3.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))

import _cub                                                  # noqa: E402
from logic_cbm import CUBLogicCBM                            # noqa: E402
from interpret_recttree_cub import load_species_names        # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "saved",
                                                   "logiccbm_k312_250n_inception_v3.pt"))
    ap.add_argument("--k", type=int, default=312)
    ap.add_argument("--n-neurons", type=int, default=250)
    ap.add_argument("--backbone", default="inception_v3")
    ap.add_argument("--focus", type=int, default=88, help="class idx to report (88=Hooded Merganser)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cpu")   # keep GPU free for the running campaign
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    stored = ck.get("acc", None) if isinstance(ck, dict) else None
    model = CUBLogicCBM(K=args.k, n_neurons=args.n_neurons,
                        backbone_kind=args.backbone).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if missing or unexpected:
        print(f"[load] missing={missing[:4]} unexpected={unexpected[:4]}")
    model.eval()
    smsg = f"  stored acc={stored*100:.2f}%" if stored is not None else ""
    print(f"loaded {os.path.basename(args.ckpt)}{smsg}", flush=True)

    image_size = 299 if args.backbone == "inception_v3" else 224
    vl = DataLoader(_cub._CUBImages("test", False, attr312=(args.k == 312),
                                    image_size=image_size),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=False)

    n_cls = 200
    correct = torch.zeros(n_cls)
    total = torch.zeros(n_cls)
    overall = tot = 0
    for img, _c, y in vl:
        logits, _ = model(img.to(device))
        pred = logits.argmax(1).cpu()
        for t, p in zip(y, pred):
            total[int(t)] += 1
            correct[int(t)] += int(p == t)
        overall += int((pred == y).sum())
        tot += y.size(0)

    sp = load_species_names()
    acc = overall / tot
    recall = (correct / total.clamp(min=1))
    print(f"\noverall test acc = {acc*100:.2f}%  ({overall}/{tot})"
          f"   [checkpoint stored {stored*100:.2f}%]" if stored else
          f"\noverall test acc = {acc*100:.2f}%")
    h = args.focus
    print(f"\nFOCUS class {h} = {sp[h]}: recall {recall[h]*100:.1f}%  "
          f"({int(correct[h])}/{int(total[h])} test images)")

    order = torch.argsort(recall)
    print("\n10 hardest classes (lowest recall):")
    for idx in order[:10]:
        i = int(idx)
        print(f"  s{i:>3} {sp[i]:<30} recall {recall[i]*100:5.1f}%  "
              f"({int(correct[i])}/{int(total[i])})")


if __name__ == "__main__":
    main()
