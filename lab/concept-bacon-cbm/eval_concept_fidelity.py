r"""Concept-fidelity / leakage check for recttree or joint checkpoints across
datasets (CIFAR-100 / AwA2 / CUB). Measures whether the (possibly fine-tuned)
concept encoder still predicts the CLASS-level concepts, via a threshold-free
nearest-class-signature accuracy. Low signature acc with high class acc => the
tree reads leaked sub-threshold structure rather than the labeled concepts.

    CUDA_VISIBLE_DEVICES="" py -3 eval_concept_fidelity.py --dataset awa \
        --ckpt saved/awa_recttree_w64_l00.0001_bin_s0_inception_v3_200ep.pt --n 1500
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


def _dataset(name, size):
    if name == "cifar":
        import cifar_data
        return cifar_data._CIFARImages("test", False, image_size=size), \
            cifar_data.load_class_attr(), 925
    if name == "awa":
        import awa_data
        return awa_data._AwAImages("test", image_size=size), \
            awa_data.load_class_attr(), 85
    if name == "cub":
        return _cub._CUBImages("test", False, attr312=True, image_size=size), \
            _cub.load_class_attr_312(), 312
    raise ValueError(name)


def _load_encoder(ckpt, K, backbone_kind):
    backbone, feat_dim, size = _cub.make_backbone(backbone_kind)
    concept = torch.nn.Linear(feat_dim, K)
    ck = torch.load(ckpt, map_location="cpu")
    if isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        cc = {k[len("concept."):]: v for k, v in sd.items() if k.startswith("concept.")}
    elif isinstance(ck, dict) and "backbone" in ck:
        bb, cc = ck["backbone"], ck["concept"]
    else:                                                      # raw state_dict
        bb = {k[len("backbone."):]: v for k, v in ck.items() if k.startswith("backbone.")}
        cc = {k[len("concept."):]: v for k, v in ck.items() if k.startswith("concept.")}
    mb, ub = backbone.load_state_dict(bb, strict=False)
    mc, uc = concept.load_state_dict(cc, strict=False)
    mb = [m for m in mb if "num_batches_tracked" not in m]
    backbone.eval(); concept.eval()
    return backbone, concept, size, (len(mb), len(ub), len(mc), len(uc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar", "awa", "cub"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--backbone", default="inception_v3")
    args = ap.parse_args()

    # need K first; peek dataset with size guess then rebuild loader with real size
    backbone, concept, size, load_info = _load_encoder(args.ckpt, {"cifar": 925, "awa": 85, "cub": 312}[args.dataset], args.backbone)
    test, class_attr, K = _dataset(args.dataset, size)
    print(f"[{args.dataset}] loaded {os.path.basename(args.ckpt)}  K={K}  "
          f"(miss/unexp bb {load_info[0]}/{load_info[1]} concept {load_info[2]}/{load_info[3]})")

    ca_norm = class_attr / class_attr.norm(dim=1, keepdim=True).clamp_min(1e-6)
    n = min(args.n, len(test))
    loader = DataLoader(Subset(test, list(range(n))), batch_size=32, num_workers=2)

    TP = FP = FN = 0
    bit_correct = bit_total = 0
    sig_correct = 0
    with torch.no_grad():
        for img, c, y in loader:
            logit = concept(backbone(img))
            probs = torch.sigmoid(logit)
            pred = (logit > 0).float()
            TP += int(((pred == 1) & (c == 1)).sum())
            FP += int(((pred == 1) & (c == 0)).sum())
            FN += int(((pred == 0) & (c == 1)).sum())
            bit_correct += int((pred == c).sum()); bit_total += c.numel()
            pn = probs / probs.norm(dim=1, keepdim=True).clamp_min(1e-6)
            sig_correct += int(((pn @ ca_norm.t()).argmax(1) == y).sum())

    recall = TP / max(TP + FN, 1)
    precision = TP / max(TP + FP, 1)
    bitacc = bit_correct / max(bit_total, 1)
    pos_rate = (TP + FN) / max(bit_total, 1)
    sig_acc = sig_correct / n
    chance = 100.0 / class_attr.shape[0]
    print(f"  active-concept fraction  : {pos_rate*100:.2f}%")
    print(f"  active-concept recall    : {recall*100:.2f}%")
    print(f"  active-concept precision : {precision*100:.2f}%")
    print(f"  per-bit accuracy         : {bitacc*100:.2f}%")
    print(f"  NEAREST-signature acc    : {sig_acc*100:.2f}%   (chance {chance:.2f}%)  <- leakage test")


if __name__ == "__main__":
    main()
