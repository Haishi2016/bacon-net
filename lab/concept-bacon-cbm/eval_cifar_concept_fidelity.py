r"""CIFAR-100 concept-fidelity / leakage check for a recttree tree checkpoint.

Loads only the backbone + concept encoder from the tree checkpoint and measures
how well the fine-tuned encoder still predicts the 925 CLASS-LEVEL concepts on
the CIFAR-100 test set. High active-concept recall => the 84.7% class accuracy is
a faithful through-the-bottleneck result; low recall (with high class acc) =>
concept leakage (the bottleneck encodes class info beyond its labels).

    CUDA_VISIBLE_DEVICES="" py -3 eval_cifar_concept_fidelity.py \
        --ckpt saved/cifar_recttree_w64_l00.0001_bin_s0_inception_v3_200ep.pt --n 2000
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))

import _cub                                                    # noqa: E402
import cifar_data                                              # noqa: E402

_K = 925


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000, help="test-subset size")
    ap.add_argument("--backbone", default="inception_v3")
    args = ap.parse_args()

    device = torch.device("cpu")
    backbone, feat_dim, size = _cub.make_backbone(args.backbone)
    concept = torch.nn.Linear(feat_dim, _K)

    sd_ck = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in sd_ck:                                  # tree checkpoint
        sd = sd_ck["state_dict"]
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        cc = {k[len("concept."):]: v for k, v in sd.items() if k.startswith("concept.")}
    else:                                                      # joint checkpoint
        bb, cc = sd_ck["backbone"], sd_ck["concept"]
    mb, ub = backbone.load_state_dict(bb, strict=False)
    mc, uc = concept.load_state_dict(cc, strict=False)
    mb = [m for m in mb if "num_batches_tracked" not in m]
    print(f"loaded {os.path.basename(args.ckpt)}  backbone(miss {len(mb)}/unexp {len(ub)}) "
          f"concept(miss {len(mc)}/unexp {len(uc)})")
    backbone.eval(); concept.eval()

    test = cifar_data._CIFARImages("test", False, image_size=size)
    n = min(args.n, len(test))
    loader = DataLoader(Subset(test, list(range(n))), batch_size=32, num_workers=2)

    class_attr = cifar_data.load_class_attr()                  # (100, 925) binary signatures
    # z-normalize signatures for a fair nearest-signature (cosine) match
    ca = class_attr
    ca_norm = ca / ca.norm(dim=1, keepdim=True).clamp_min(1e-6)

    TP = FP = FN = TN = 0
    bit_correct = bit_total = 0
    sig_correct = sig_total = 0
    with torch.no_grad():
        for img, c, y in loader:
            logit = concept(backbone(img))
            probs = torch.sigmoid(logit)
            pred = (logit > 0).float()
            TP += int(((pred == 1) & (c == 1)).sum())
            FP += int(((pred == 1) & (c == 0)).sum())
            FN += int(((pred == 0) & (c == 1)).sum())
            TN += int(((pred == 0) & (c == 0)).sum())
            bit_correct += int((pred == c).sum()); bit_total += c.numel()
            # nearest class-signature (cosine) from the CONTINUOUS concept probs
            pn = probs / probs.norm(dim=1, keepdim=True).clamp_min(1e-6)
            sig_pred = (pn @ ca_norm.t()).argmax(1)
            sig_correct += int((sig_pred == y).sum()); sig_total += len(y)

    recall = TP / max(TP + FN, 1)
    precision = TP / max(TP + FP, 1)
    bitacc = bit_correct / max(bit_total, 1)
    pos_rate = (TP + FN) / max(bit_total, 1)                    # active-concept fraction
    sig_acc = sig_correct / max(sig_total, 1)
    print(f"\nCONCEPT FIDELITY on {n} CIFAR-100 test images  (K={_K})")
    print(f"  active-concept fraction (label)   : {pos_rate*100:.2f}%  (~{pos_rate*_K:.0f}/{_K} per image)")
    print(f"  active-concept RECALL             : {recall*100:.2f}%")
    print(f"  active-concept precision          : {precision*100:.2f}%  (pos_weight biases toward positives)")
    print(f"  overall per-bit accuracy          : {bitacc*100:.2f}%")
    print(f"\n  NEAREST class-signature accuracy  : {sig_acc*100:.2f}%   <- threshold-free leakage test")
    print(f"  (predict class = argmax cosine(predicted concept vector, class's 925-bit signature))")
    print(f"\n  interpretation: if nearest-signature acc ~= the tree's 84.7%, the concept vector "
          f"faithfully encodes the class via its LABELED concepts; if it is far lower, the tree "
          f"is reading sub-threshold structure beyond the concept labels (leakage).")


if __name__ == "__main__":
    main()
