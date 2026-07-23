"""
Zero-shot comparison: BaconCBM vs CREAM on synthetic shapes (MNIST digit models).

Both models are trained on MNIST DIGIT LABELS ONLY (no stroke-concept
supervision), share the same 9 human stroke concepts, and the same per-digit
reasoning (which concepts each digit depends on):

  * BaconCBM : fixed AND/OR/NOT logic trees over the concepts (frozen structure).
  * CREAM    : structured C->Y (a masked linear that lets each digit see only its
               rule's concepts) -- the CBM analogue of the same reasoning graph.

We then apply both, zero-shot, to synthetic shapes and ask: whose concepts +
digit-0 head transfer to circle detection?  This tests whether BACON's rigid
logic yields more human-aligned / transferable concepts than CREAM's trained
masked linear given the same supervision.

    python eval_cream_zeroshot.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg                                        # noqa: E402
import shapes as shapes_mod                                 # noqa: E402
from bacon_logic import parse_formula, collect_concepts     # noqa: E402
from model import ConceptCNN                                # noqa: E402
from eval_shapes import load_model, roc_auc                 # noqa: E402
from train import make_loaders                              # noqa: E402


# --------------------------------------------------------------------------- #
# CREAM digit model: same concepts + reasoning graph, trained masked C->Y head
# --------------------------------------------------------------------------- #
def build_AY(concept_names, rules):
    """A_Y (10, K): digit -> concepts referenced in its rule (the reasoning graph)."""
    idx = {n: i for i, n in enumerate(concept_names)}
    A = torch.zeros(len(rules), len(concept_names))
    for d, formula in rules.items():
        for name in collect_concepts(parse_formula(formula), set()):
            A[d, idx[name]] = 1.0
    return A


class MaskedLinear(nn.Module):
    def __init__(self, in_f, out_f, mask):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        return F.linear(x, self.lin.weight * self.mask, self.lin.bias)


class CREAMDigit(nn.Module):
    """CBM with the same concepts + reasoning graph, masked C->Y (no side-channel
    by default = fair vs BaconCBM which also has no side-channel)."""

    def __init__(self, concept_names, rules, d_y=0, dropout_p=0.9):
        super().__init__()
        self.concept_names = list(concept_names)
        K, L = len(concept_names), len(rules)
        cnn = ConceptCNN(K)
        self.features = cnn.features
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3))
        self.d_y = d_y
        self.splitter = nn.Linear(128, K + d_y)
        A_Y = build_AY(concept_names, rules)                 # (L,K)
        if d_y > 0:
            self.p = dropout_p
            self.side = nn.Linear(d_y, L)
            mask = torch.cat([A_Y, torch.eye(L)], dim=1)     # (L, K+L)
            self.task = MaskedLinear(K + L, L, mask)
        else:
            self.task = MaskedLinear(K, L, A_Y)               # (L,K)

    def concept_probs(self, x):
        return torch.sigmoid(self.splitter(self.fc(self.features(x)))[:, :len(self.concept_names)])

    def forward(self, x):
        z = self.splitter(self.fc(self.features(x)))
        K = len(self.concept_names)
        c = torch.sigmoid(z[:, :K])
        if self.d_y > 0:
            zy = z[:, K:]
            if self.training:
                keep = (torch.rand(zy.shape[0], 1, device=zy.device) > self.p).float()
                zy = zy * keep
            logits = self.task(torch.cat([c, self.side(zy)], dim=1))
        else:
            logits = self.task(c)
        return logits, c


def train_cream(model, tl, device, epochs):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def mnist_acc(model, vl, device, bacon=False):
    model.eval(); correct = total = 0
    for x, y in vl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0]
        correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
    return correct / total


@torch.no_grad()
def scores_and_concepts(model, imgs, is_bacon):
    """Return (circle_score, concept_probs) for a batch, model-agnostic."""
    out = model(imgs)
    if is_bacon:
        _, probs, truths = out
        return truths[:, 0], probs            # "0" tree truth
    logits, c = out
    return torch.softmax(logits, 1)[:, 0], c  # P(digit 0)


def _auc_and_gap(model, is_bacon, imgs_by_cat, round_set, ci):
    """Circle-score AUC (round vs non-round) + loop-concept alignment gap.

    The alignment gap = mean(loop concepts | round) - mean(loop | non-round).
    A human-aligned encoder makes loop_upper/loop_lower fire on round objects
    and stay low on non-round ones (large positive gap).
    """
    lu, ll = ci["loop_upper"], ci["loop_lower"]
    scores, labels, loop_round, loop_non = [], [], [], []
    for cat, imgs in imgs_by_cat.items():
        s, probs = scores_and_concepts(model, imgs, is_bacon)
        is_round = 1 if cat in round_set else 0
        scores.append(s.cpu())
        labels.append(torch.full((len(s),), is_round))
        loop = 0.5 * (probs[:, lu] + probs[:, ll]).mean().item()
        (loop_round if is_round else loop_non).append(loop)
    auc = roc_auc(torch.cat(scores), torch.cat(labels))
    mr = sum(loop_round) / max(len(loop_round), 1)
    mn = sum(loop_non) / max(len(loop_non), 1)
    return auc, mr, mn


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*",
                    default=["shapes", "quickdraw", "photos"],
                    help="which zero-shot domains to run")
    ap.add_argument("--n", type=int, default=400, help="images per category")
    args = ap.parse_args()
    domains = set(args.domains)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = os.path.join(_HERE, "..", "..", "benchmarks", "mnist-addition", "data")
    tl, vl = make_loaders(data, 256)

    # BaconCBM (existing checkpoint)
    ckpt = os.path.join(_HERE, "checkpoint.pt")
    bacon = load_model(ckpt, device)
    bacon.eval()

    # CREAM digit model (train, cache)
    cream_path = os.path.join(_HERE, "cream_digit.pt")
    cream = CREAMDigit(cfg.CONCEPTS, {int(k): v for k, v in cfg.DIGIT_RULES.items()}, d_y=0)
    if os.path.exists(cream_path):
        cream.load_state_dict(torch.load(cream_path, map_location=device))
        cream.to(device)
    else:
        print("training CREAM digit model (task-only, no concept supervision)...")
        train_cream(cream, tl, device, epochs=6)
        torch.save(cream.state_dict(), cream_path)
    cream.eval()

    print(f"\nMNIST test acc:  BaconCBM {mnist_acc(bacon, vl, device)*100:.2f}  "
          f"CREAM {mnist_acc(cream, vl, device)*100:.2f}")

    # ---- zero-shot on synthetic shapes ---- #
    imgs, names = shapes_mod.generate(400, seed=123)
    imgs = imgs.to(device)
    ci = {n: i for i, n in enumerate(cfg.CONCEPTS)}
    lu, ll, vl_i = ci["loop_upper"], ci["loop_lower"], ci["vertical_line"]

    with torch.no_grad():
        _, bp, btr = bacon(imgs)                      # bacon: (logits, probs, truths)
        b_score = btr[:, 0]                           # "0" tree truth
        cl, cp = cream(imgs)                          # cream: (logits, concepts)
        c_score = torch.softmax(cl, 1)[:, 0]          # P(digit 0)

    labels = torch.tensor([1 if n in shapes_mod.ROUND_SHAPES else 0 for n in names])
    strict = [i for i, n in enumerate(names) if n == "circle" or n not in shapes_mod.ROUND_SHAPES]
    s_lab = torch.tensor([1 if names[i] == "circle" else 0 for i in strict])

    print("\nZero-shot circle detection (digit-0 score):")
    print(f"{'model':10s} {'round-vs-rest AUC':>18s} {'circle-vs-poly/line AUC':>24s}")
    for tag, s in [("BaconCBM", b_score.cpu()), ("CREAM", c_score.cpu())]:
        auc_all = roc_auc(s, labels)
        auc_strict = roc_auc(s[strict], s_lab)
        print(f"{tag:10s} {auc_all:18.3f} {auc_strict:24.3f}")

    print("\nConcept activation on circles vs lines (loopU / loopL / vertical):")
    for tag, probs in [("BaconCBM", bp.cpu()), ("CREAM", cp.cpu())]:
        ci_c = [i for i, n in enumerate(names) if n == "circle"]
        ci_l = [i for i, n in enumerate(names) if n == "line"]
        cir = probs[ci_c].mean(0); lin = probs[ci_l].mean(0)
        print(f"  {tag:9s} circle: {cir[lu]:.2f}/{cir[ll]:.2f}/{cir[vl_i]:.2f}   "
              f"line: {lin[lu]:.2f}/{lin[ll]:.2f}/{lin[vl_i]:.2f}")

    models = [("BaconCBM", bacon, True), ("CREAM", cream, False)]
    hdr = (f"{'model':10s} {'round-vs-nonround AUC':>22s} "
           f"{'loop(round)':>12s} {'loop(non)':>10s} {'align-gap':>10s}")

    # ---- zero-shot on QuickDraw everyday-object doodles ---- #
    if "quickdraw" in domains:
        import quickdraw as qd
        ROUND = ["circle", "donut", "clock", "wheel", "cookie", "basketball", "pizza"]
        NON_ROUND = ["ladder", "envelope", "line", "pants", "table", "fork",
                     "pencil", "zigzag"]
        round_set = set(ROUND)
        qd_imgs = {}
        for cat in ROUND + NON_ROUND:
            try:
                qd_imgs[cat] = qd.load_category(cat, n=args.n).to(device)
            except Exception as e:
                print(f"  skip quickdraw:{cat} ({e})")
        print("\n=== QuickDraw everyday objects (zero-shot, real doodles) ===")
        print(hdr)
        for tag, model, is_bacon in models:
            auc, mr, mn = _auc_and_gap(model, is_bacon, qd_imgs, round_set, ci)
            print(f"{tag:10s} {auc:22.3f} {mr:12.2f} {mn:10.2f} {mr - mn:10.2f}")

    # ---- zero-shot on real object photos (edge sketch) ---- #
    if "photos" in domains:
        import photo_sketch
        import eval_photos as ep
        spec = ep.DATASETS["caltech101"]
        cache = os.path.join(_HERE, "cache")
        ep.ensure_dataset(spec, cache)
        round_set = set(spec["round"])
        nph = min(args.n, 120)
        ph_imgs = {}
        for cat in spec["round"] + spec["non_round"]:
            try:
                raw = ep.load_class(spec, cache, cat, nph).to(device)
                ph_imgs[cat] = photo_sketch.to_sketch(raw, blur=2, keep_frac=0.12)
            except Exception as e:
                print(f"  skip photo:{cat} ({e})")
        print("\n=== Caltech-101 real photos -> edge sketch (zero-shot) ===")
        print(hdr)
        for tag, model, is_bacon in models:
            auc, mr, mn = _auc_and_gap(model, is_bacon, ph_imgs, round_set, ci)
            print(f"{tag:10s} {auc:22.3f} {mr:12.2f} {mn:10.2f} {mr - mn:10.2f}")


if __name__ == "__main__":
    main()
