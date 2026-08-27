"""Zero-shot species accuracy on NABirds for the CUB head-ablation models.

Validity check for the CUB->NABirds faithfulness comparison: if a model cannot
classify the mapped NABirds birds above chance, its concept-faithfulness number
is not meaningful. Loads the saved encoders (no retraining) and reports CUB test
accuracy and NABirds zero-shot species accuracy (predict the mapped CUB species).

    python eval_nabirds_task.py --ontology saved/cub_tree_k24_800ep_harden.pt \
        --mlp saved/cub_bb_mlp.pt --linear saved/cub_bb_linear.pt \
        --cbm saved/cub_cbm_sup.pt --K 24 --nabirds C:\\School\\datasets\\nabirds
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
import nabirds                                                  # noqa: E402
from cub_emergent import CUBEmergent, evaluate                  # noqa: E402
from ablation_head_cub import CUBBlackBox                       # noqa: E402


def load_ontology(path, K, device):
    m = CUBEmergent(K, head="tree").to(device)
    m.load_state_dict(torch.load(path, map_location=device)["state_dict"],
                      strict=False)
    m.eval()
    return m


def load_bb(path, device):
    ck = torch.load(path, map_location=device)
    m = CUBBlackBox(ck["K"], head=ck["head"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default=None)
    ap.add_argument("--mlp", default=None)
    ap.add_argument("--linear", default=None)
    ap.add_argument("--cbm", default=None)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--nabirds", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    na = DataLoader(nabirds.NABirdsCUB(args.nabirds), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)

    models = []
    if args.ontology and os.path.exists(args.ontology):
        models.append(("ontology (OCBM)", load_ontology(args.ontology, args.K, device)))
    if args.mlp and os.path.exists(args.mlp):
        models.append(("mlp (task-only)", load_bb(args.mlp, device)))
    if args.linear and os.path.exists(args.linear):
        models.append(("linear (task-only)", load_bb(args.linear, device)))
    if args.cbm and os.path.exists(args.cbm):
        models.append(("cbm (K=112, sup)", load_bb(args.cbm, device)))

    print(f"\nZERO-SHOT SPECIES ACCURACY  (CUB test vs NABirds transfer)")
    print(f"{'model':22} {'CUB acc%':>9} {'NABirds acc%':>13}")
    for name, m in models:
        cub = evaluate(m, vl, device)
        nab = evaluate(m, na, device)
        print(f"{name:22} {cub*100:8.2f} {nab*100:12.2f}")
    print("\nNABirds accuracy predicts the mapped CUB species (200-way, "
          "chance ~0.5%). If this is near chance the faithfulness comparison "
          "is not meaningful.")


if __name__ == "__main__":
    main()
