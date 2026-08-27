"""Train hardened CUB OCBM configurations for the decision-making study.

Runs (in priority order): supervised K=112 (concept BCE on all 112 attributes),
emergent K=112 (no supervision), then emergent K=96 and K=48. All 800 epochs,
hardened (Hungarian-frozen permutation -> faithful per-concept trees for the
pruning analysis). The supervised vs emergent K=112 pair is the key comparison:
same capacity, one with named-attribute concepts and one with emergent concepts.

    python run_cub_ocbm_configs.py --epochs 800
    -> saved/cub_ocbm_k{K}_{sup|emergent}_{epochs}ep.pt  + accuracy summary
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                     # noqa: E402
from cub_emergent import train_one                             # noqa: E402

# (K, supervised) in priority order -- K=120 emergent (higher-capacity scan
# point) first; the K=112/96 configs already have checkpoints and are skipped.
CONFIGS = [(120, False), (112, True), (112, False), (96, False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="concept-BCE weight for the supervised config")
    ap.add_argument("--save-dir", default=os.path.join(_HERE, "saved"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tl = DataLoader(_cub._CUBImages("train", True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    os.makedirs(args.save_dir, exist_ok=True)

    results = []
    for K, sup in CONFIGS:
        tag = f"k{K}_{'sup' if sup else 'emergent'}"
        path = os.path.join(args.save_dir, f"cub_ocbm_{tag}_{args.epochs}ep.pt")
        if os.path.exists(path):
            print(f"\n===== {tag}: checkpoint exists, SKIPPING ({path}) =====",
                  flush=True)
            continue
        print(f"\n===== {tag}  ({args.epochs}ep, hardened, "
              f"{'concept-supervised' if sup else 'no supervision'}) =====",
              flush=True)
        ct = torch.arange(112) if sup else None
        lam = args.lam if sup else 0.0
        try:
            acc, model = train_one(
                K, tl, vl, device, epochs=args.epochs, seed=args.seed,
                head="tree", loss_mode="hybrid", ovr_weight=0.3,
                harden=True, sinkhorn_iters=100, perm_sparsity=5.0,
                freeze_conf=0.90, freeze_frac=0.85, anneal_cap=1.0,
                concept_lam=lam, concept_targets=ct,
                ckpt_path=path + ".partial", ckpt_every=100)
            torch.save({"state_dict": model.state_dict(), "K": K,
                        "supervised": sup,
                        "concept_targets": (ct.tolist() if ct is not None else None)},
                       path)
            print(f"  DONE {tag}: acc {acc * 100:.2f}%  saved {path}", flush=True)
            results.append((tag, acc))
        except Exception as e:                                  # noqa: BLE001
            print(f"  {tag} FAILED: {e}", flush=True)
            traceback.print_exc()

    print("\n===== SUMMARY (CUB OCBM configs) =====", flush=True)
    for tag, acc in results:
        print(f"  {tag}: {acc * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
