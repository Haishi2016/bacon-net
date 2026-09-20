"""Verify a saved hardened OCBM fulltree checkpoint reloads and reproduces its
reported frozen accuracy on the CUB test set. Repeatability guard before we edit
train_one for the two-stage joint-pretrain experiment."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        _HERE, "saved",
        "cub_ocbm_k312_fulltree_b8_eg2_neg_sup0p03_iw_inception_v3_150ep.pt"))
    args = ap.parse_args()
    ckpt = args.ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    meta = {k: ck[k] for k in ck if k != "state_dict"}
    print("checkpoint meta:", meta, flush=True)

    model = CUBEmergent(ck["K"], head=ck["head"], branching=ck["branching"],
                        fulltree_negation=ck.get("negation", False),
                        fulltree_max_egress=ck.get("max_egress", 1),
                        backbone_kind=ck["backbone"]).to(device)
    # allocate the frozen egress buffers + switch the head to the hardened path so
    # the trained frozen structure loads and forward evaluates the discrete tree.
    model.head.freeze_egress()
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    print(f"load: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    if missing:
        print("  missing[:8]:", list(missing)[:8], flush=True)
    if unexpected:
        print("  unexpected[:8]:", list(unexpected)[:8], flush=True)
    model.eval()

    image_size = 299 if ck["backbone"] == "inception_v3" else 224
    vl = DataLoader(_cub._CUBImages("test", False, attr312=ck.get("attr312", False),
                                    image_size=image_size),
                    batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    acc = evaluate(model, vl, device)
    print(f"RELOAD FROZEN EVAL ACC = {acc * 100:.2f}%   ({os.path.basename(ckpt)})", flush=True)


if __name__ == "__main__":
    main()
