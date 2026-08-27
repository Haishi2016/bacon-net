"""Train an EMERGENT K=112 CUB OCBM with a concept-activation DECORRELATION
penalty, to test whether pushing concepts onto distinct attributes removes the
redundancy collapse (5x back_color::yellow) seen in the plain emergent model.

Everything else matches the plain emergent K=112 config (800ep, hybrid loss,
hardened -> faithful frozen trees) so the only difference is decorr_lam.
Saves to a DISTINCT path so the plain emergent checkpoint is untouched.

    py -3 -u run_cub_decorr.py --epochs 800 --decorr-lam 1.0
    -> saved/cub_ocbm_k112_emergent_decorr{lam}_800ep.pt
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
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                     # noqa: E402
from cub_emergent import train_one                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=112)
    ap.add_argument("--decorr-lam", type=float, default=1.0)
    ap.add_argument("--save-dir", default=os.path.join(_HERE, "saved"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tl = DataLoader(_cub._CUBImages("train", True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    os.makedirs(args.save_dir, exist_ok=True)

    lam_tag = f"{args.decorr_lam:g}".replace(".", "p")
    path = os.path.join(args.save_dir,
                        f"cub_ocbm_k{args.K}_emergent_decorr{lam_tag}_{args.epochs}ep.pt")
    if os.path.exists(path):
        print(f"checkpoint exists, SKIPPING ({path})", flush=True)
        return

    print(f"\n===== k{args.K}_emergent_decorr (lam={args.decorr_lam}, "
          f"{args.epochs}ep, hardened) =====", flush=True)
    acc, model = train_one(
        args.K, tl, vl, device, epochs=args.epochs, seed=args.seed,
        head="tree", loss_mode="hybrid", ovr_weight=0.3,
        harden=True, sinkhorn_iters=100, perm_sparsity=5.0,
        freeze_conf=0.90, freeze_frac=0.85, anneal_cap=1.0,
        concept_lam=0.0, concept_targets=None,
        decorr_lam=args.decorr_lam,
        ckpt_path=path + ".partial", ckpt_every=100)
    torch.save({"state_dict": model.state_dict(), "K": args.K,
                "supervised": False, "decorr_lam": args.decorr_lam}, path)
    print(f"  DONE k{args.K}_emergent_decorr: acc {acc * 100:.2f}%  saved {path}",
          flush=True)


if __name__ == "__main__":
    main()
