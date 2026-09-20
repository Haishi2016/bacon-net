"""Test-time augmentation (horizontal flip) eval for a hardened OCBM fulltree
checkpoint. CUB birds are roughly left-right symmetric, so averaging the class
logits over the image and its mirror is a near-free accuracy bump and measures
how much signal the frozen concepts already carry (headroom check before we
retrain the backbone). No retraining, loads the exact frozen structure."""
import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                    # noqa: E402
from cub_emergent import CUBEmergent, evaluate                 # noqa: E402


@torch.no_grad()
def evaluate_tta(model, loader, device):
    """Average class logits over the original and horizontally-flipped image."""
    model.eval()
    correct = total = 0
    for img, c, y in loader:
        img, y = img.to(device), y.to(device)
        logits = model(img)[0]
        logits_f = model(torch.flip(img, dims=[3]))[0]
        pred = (logits + logits_f).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        _HERE, "saved",
        "cub_ocbm_k312_fulltree_b8_eg2_neg_sup0p03_iw_pt_cwovr1_inception_v3_150ep.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = CUBEmergent(ck["K"], head=ck["head"], branching=ck["branching"],
                        fulltree_negation=ck.get("negation", False),
                        fulltree_max_egress=ck.get("max_egress", 1),
                        backbone_kind=ck["backbone"]).to(device)
    model.head.freeze_egress()
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    print(f"load: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    model.eval()

    image_size = 299 if ck["backbone"] == "inception_v3" else 224
    vl = DataLoader(_cub._CUBImages("test", False, attr312=ck.get("attr312", False),
                                    image_size=image_size),
                    batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    plain = evaluate(model, vl, device)
    tta = evaluate_tta(model, vl, device)
    print(f"PLAIN  EVAL ACC = {plain * 100:.2f}%", flush=True)
    print(f"TTA(hflip) ACC  = {tta * 100:.2f}%   (+{(tta - plain) * 100:.2f}pt)   "
          f"({os.path.basename(args.ckpt)})", flush=True)


if __name__ == "__main__":
    main()
