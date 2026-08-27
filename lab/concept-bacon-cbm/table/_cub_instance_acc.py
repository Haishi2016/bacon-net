"""
Data-only check: how much do CUB CLASS-MAJORITY concept labels (Koh pkls, what we
train/eval against) differ from INSTANCE-level per-image annotations
(attributes/image_attribute_labels.txt)?

A model whose concept predictions track the class-majority labels (our ACC_C ~93)
would, if scored against instance labels, cap near this agreement.  If that is
~86, it explains/justifies the paper's lower ACC_C without any model defect.

    python _cub_instance_acc.py
"""

from __future__ import annotations

import os
import pickle

import torch

CUB = r"C:\School\datasets\cub\CUB_200_2011"

# 0-based indices of the canonical 112 concepts into the 312 raw attributes.
CANONICAL_112 = (
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101,
    106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152,
    153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194,
    196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236,
    238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277,
    283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311,
)
_RAW = 312


def load_instance(image_ids):
    """img_id -> (present(312), visible(312)) from image_attribute_labels.txt."""
    path = os.path.join(CUB, "attributes", "image_attribute_labels.txt")
    ids = set(image_ids)
    present = {i: torch.zeros(_RAW) for i in ids}
    visible = {i: torch.zeros(_RAW) for i in ids}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            img_id, attr_id, is_present, certainty = (int(p[0]), int(p[1]),
                                                      int(p[2]), int(p[3]))
            if img_id in present:
                present[img_id][attr_id - 1] = float(is_present)
                visible[img_id][attr_id - 1] = 0.0 if certainty == 1 else 1.0
    return present, visible


def main():
    with open(os.path.join(CUB, "test.pkl"), "rb") as f:
        entries = pickle.load(f)
    ids = [e["id"] for e in entries]
    # class-majority concept vector per image (what our ACC_C scores against)
    cm = torch.stack([torch.tensor(e["attribute_label"], dtype=torch.float32)
                      for e in entries])                      # (N, 112)

    present, visible = load_instance(ids)
    sel = list(CANONICAL_112)
    inst = torch.stack([present[i][sel] for i in ids])         # (N, 112)
    vis = torch.stack([visible[i][sel] for i in ids])          # (N, 112)

    agree = (cm == inst).float()
    print(f"test images: {len(ids)}   concepts: {cm.shape[1]}")
    print(f"class-majority vs instance agreement (all)      : "
          f"{agree.mean().item() * 100:.2f}%")
    v = vis.bool()
    print(f"class-majority vs instance agreement (visible)  : "
          f"{agree[v].mean().item() * 100:.2f}%")
    print(f"  (fraction of (img,concept) marked visible: {v.float().mean() * 100:.1f}%)")
    print(f"instance positive rate: {inst.mean().item() * 100:.1f}%   "
          f"class-majority positive rate: {cm.mean().item() * 100:.1f}%")


if __name__ == "__main__":
    main()
