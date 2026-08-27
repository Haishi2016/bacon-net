"""Zero-shot concept transfer test: does each emergent concept survive on an
EXTERNAL handwritten-digit corpus (USPS) the encoder never saw?

If c2 is a genuine "4" concept, its 4-vs-rest AUC should stay high on USPS.
If c2 is an MNIST-specific shortcut (its near-zero occlusion saliency is a
warning), its AUC should collapse while a real primitive like c1 (closed-loop,
0/8) survives the domain shift.

We run the FROZEN MNIST encoder on:
  * MNIST test  (in-domain reference)
  * USPS test   (out-of-domain, resized 16x16 -> 28x28, MNIST-normalized)
and report, per concept, the best one-vs-rest digit and that digit's AUC on
each corpus, plus the transfer gap.

    python eval_concept_transfer.py --load saved/k5_harden.pt

Hypothesized targets (from concept_receptive_fields):
    c0->{5,9}  c1->{8,0}(loop)  c2->{4}  c3->{2}  c4->{7,4}
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import urllib.request

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def _install_verified_opener():
    """Verify certs against certifi's trust store (chain + hostname stay ON),
    but clear only VERIFY_X509_STRICT, which Python 3.14 enables by default and
    which some legacy dataset hosts fail ('Missing Subject Key Identifier').
    We do NOT disable certificate verification."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from train_emergent_concepts import MultiTreeBaconCBM  # noqa: E402
from eval_shapes import roc_auc                         # noqa: E402

_MEAN, _STD = 0.1307, 0.3081
# claimed family targets per concept (for the "does the target survive" summary)
TARGETS = {0: [5, 9], 1: [8, 0], 2: [4], 3: [2], 4: [7, 4]}


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = MultiTreeBaconCBM(ckpt["K"], weight_mode=ckpt.get("weight_mode", "trainable"),
                          no_negation=ckpt.get("no_negation", False),
                          device=device).to(device)
    if ckpt.get("frozen"):
        m.prepare_frozen_structure(); m.load_state_dict(ckpt["state_dict"])
    else:
        m.load_state_dict(ckpt["state_dict"]); m.anneal(1.0)
    m.eval()
    return m, ckpt


def mnist_loader(root, bs):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((_MEAN,), (_STD,))])
    ds = datasets.MNIST(root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs)


def usps_loader(root, bs):
    # USPS is 16x16 grayscale; resize to 28x28 and apply MNIST normalization so
    # the encoder sees the same statistics it was trained on.
    _install_verified_opener()
    tf = transforms.Compose([transforms.Resize(28), transforms.ToTensor(),
                             transforms.Normalize((_MEAN,), (_STD,))])
    ds = datasets.USPS(root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs)


@torch.no_grad()
def collect(model, loader, device):
    cs, ys = [], []
    for x, y in loader:
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(cs), torch.cat(ys)


def per_digit_auc(concept_scores, y, ci):
    """AUC of concept ci for each digit d as one-vs-rest positive."""
    return {d: roc_auc(concept_scores[:, ci], (y == d).long()) for d in range(10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.load, device)
    K = ckpt["K"]

    Cm, Ym = collect(model, mnist_loader(args.data, args.batch_size), device)
    Cu, Yu = collect(model, usps_loader(args.data, args.batch_size), device)
    print(f"\nCONCEPT TRANSFER  {os.path.basename(args.load)}  K={K}"
          f"   MNIST N={len(Ym)}   USPS N={len(Yu)}")

    print("\nper-concept best one-vs-rest digit  (AUC>0.5 means concept selects it):")
    print(f"{'':4} {'MNIST best':>22} {'USPS best':>22}   {'target d':>8} "
          f"{'MNIST':>7} {'USPS':>7} {'gap':>6}")
    for i in range(K):
        am = per_digit_auc(Cm, Ym, i)
        au = per_digit_auc(Cu, Yu, i)
        bm_d = max(am, key=am.get); bu_d = max(au, key=au.get)
        # target-family AUC = best AUC over the hypothesized target digits
        tgt = TARGETS.get(i, [bm_d])
        tm = max(am[d] for d in tgt); tu = max(au[d] for d in tgt)
        td = tgt[int(torch.tensor([am[d] for d in tgt]).argmax())]
        gap = tm - tu
        flag = ""
        if tu < 0.65:
            flag = "  <-- collapses (shortcut)"
        elif gap < 0.10 and tu >= 0.75:
            flag = "  <-- transfers"
        print(f"c{i}: MNIST d{bm_d}={am[bm_d]:.2f}  |  USPS d{bu_d}={au[bu_d]:.2f}"
              f"   d{td}    {tm:.2f}   {tu:.2f}  {gap:+.2f}{flag}")

    # focused c2 = "4" report
    if 2 < K:
        a4m = roc_auc(Cm[:, 2], (Ym == 4).long())
        a4u = roc_auc(Cu[:, 2], (Yu == 4).long())
        print(f"\nc2 as a 4-detector:  MNIST 4-vs-rest AUC={a4m:.3f}   "
              f"USPS 4-vs-rest AUC={a4u:.3f}   gap={a4m - a4u:+.3f}")
        print("  (near-0 occlusion saliency predicted this would NOT transfer if"
              " it is a shortcut.)" if a4u < 0.65 else
              "  (survives the domain shift => genuine transferable 4 concept.)")


if __name__ == "__main__":
    main()
