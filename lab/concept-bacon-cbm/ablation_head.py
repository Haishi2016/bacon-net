"""Key ablation: does the graded-logic ONTOLOGY cause human-aligned concepts?

Holds encoder, K, and task-only supervision fixed; swaps only the reasoning
layer: OCBM graded-logic ontology vs. a black-box MLP head vs. a linear head.
Trains all three from scratch, MULTI-SEED, and reports task accuracy plus
concept-quality metrics computed on each model's K-dim sigmoid bottleneck:

  * named-AUC (alignment): for each emergent concept, its best identification
    AUC against the 9 human stroke concepts (loop_upper, vertical_line, ...),
    using the per-digit rule targets as ground truth (mean over concepts). This
    measures whether each emergent axis corresponds to a human primitive.
  * coverage: for each human stroke, the best-matching emergent concept's AUC
    (mean over strokes) -- whether the human primitives are represented.
  * distinctness (mean |R|), faithfulness (MNIST->USPS best-digit AUC gap).

    python ablation_head.py --K 5 --epochs 20 --seeds 3
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from model import ConceptCNN                                    # noqa: E402
from train_emergent_concepts import make_loaders, train_one     # noqa: E402
from concept_correlations import corr_matrix                    # noqa: E402
from concept_receptive_fields import collect                    # noqa: E402
from eval_concept_transfer import usps_loader                   # noqa: E402
from eval_shapes import roc_auc                                 # noqa: E402
from interpret_emergent_concepts import named_gt, NAMED, nmi_binary  # noqa: E402
from eval_tree_transfer_calibrated import calibrate, predict_from_probs  # noqa: E402
import _bench                                                   # noqa: E402
from scipy.optimize import linear_sum_assignment               # noqa: E402


class BlackBoxCBM(nn.Module):
    def __init__(self, K, n_classes=10, head="mlp", hidden=64):
        super().__init__()
        self.encoder = ConceptCNN(K)
        self.head = (nn.Linear(K, n_classes) if head == "linear" else
                     nn.Sequential(nn.Linear(K, hidden), nn.ReLU(),
                                   nn.Linear(hidden, n_classes)))

    def concept_probs(self, x):
        return torch.sigmoid(self.encoder(x))

    def forward(self, x):
        c = self.concept_probs(x)
        return self.head(c), c, None


def train_blackbox(head, K, train_ld, test_ld, device, epochs, seed):
    torch.manual_seed(seed)
    m = BlackBoxCBM(K, head=head).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        m.train()
        for x, y in train_ld:
            opt.zero_grad()
            nn.functional.cross_entropy(m(x.to(device))[0], y.to(device)).backward()
            opt.step()
        sched.step()
    m.eval()
    return m


@torch.no_grad()
def accuracy(model, loader, device):
    c = t = 0
    for x, y in loader:
        p = model(x.to(device))[0].argmax(1).cpu()
        c += (p == y).sum().item(); t += len(y)
    return c / t


@torch.no_grad()
def usps_cal_acc(model, test_ld, usps_ld, device):
    """Zero-shot USPS accuracy AFTER head-agnostic per-concept quantile
    calibration (USPS concepts matched onto the model's own MNIST-test concept
    reference). Works for both the logic-tree head and a linear/mlp head."""
    Cm = torch.cat([model.concept_probs(x.to(device)).cpu() for x, _ in test_ld])
    Cu, Yu = [], []
    for x, y in usps_ld:
        Cu.append(model.concept_probs(x.to(device)).cpu()); Yu.append(y)
    Cu = torch.cat(Cu); Yu = torch.cat(Yu)
    Cu_cal = calibrate(Cu, Cm)
    if hasattr(model, "trees"):
        pred = predict_from_probs(model, Cu_cal, device)
    else:
        pred = model.head(Cu_cal.to(device)).argmax(1).cpu()
    return (pred == Yu).float().mean().item()


def train_ontology_stable(K, train_ld, test_ld, device, epochs, seed,
                          max_restarts=5, floor=0.90):
    """Train OCBM with restart-on-collapse (cold-start symmetry collapse ~1/45)."""
    acc, m = 0.0, None
    for att in range(max_restarts):
        acc, m, _ = train_one(K, train_ld, test_ld, device, epochs=epochs,
                              seed=seed + 1000 * att, verbose=False)
        if acc >= floor:
            return acc, m, att
    return acc, m, max_restarts - 1


def named_auc_matrix(C, y, spec, K):
    """K x 9 identification AUC and NMI of each concept vs each human stroke."""
    gt = named_gt(y, spec)
    A = torch.full((K, len(NAMED)), float("nan"))
    Nmi = torch.zeros(K, len(NAMED))
    for j, col in enumerate(gt):
        if col is None:
            continue
        labels, rows = col
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        for i in range(K):
            A[i, j] = roc_auc(C[rows, i], labels)
            Nmi[i, j] = nmi_binary(C[rows, i] > 0.5, labels)
    return A, Nmi


@torch.no_grad()
def concept_quality(model, test_ld, usps_ld, device, K, spec):
    X, C, Y = collect(model, test_ld, device)
    Cu, Yu = [], []
    for x, y in usps_ld:
        Cu.append(model.concept_probs(x.to(device)).cpu()); Yu.append(y)
    Cu = torch.cat(Cu); Yu = torch.cat(Yu)

    # named alignment (direction-agnostic: a concept may equal NOT a stroke)
    A, Nmi = named_auc_matrix(C, Y, spec, K)
    align = torch.nan_to_num(torch.maximum(A, 1 - A), nan=0.0)  # [0.5,1]
    named_best = align.amax(1).mean().item()                   # lenient (any stroke)
    named_cover = align.amax(0).mean().item()
    # strict: Hungarian one-to-one concept<->distinct stroke assignment
    ri, ci = linear_sum_assignment(-align.numpy())
    bij_auc = align[ri, ci].mean().item()                      # no stroke reused
    bij_nmi = Nmi[ri, ci].mean().item()

    # distinctness
    absR = corr_matrix(C).abs(); absR.fill_diagonal_(0.0)
    mean_absR = absR.mean().item()

    # faithfulness: per concept best MNIST digit, gap to USPS (selectivity)
    gaps = []
    for i in range(K):
        aucs = [roc_auc(C[:, i], (Y == d).long()) for d in range(10)]
        d = max(range(10), key=lambda dd: abs(aucs[dd] - 0.5))
        sel = abs(aucs[d] - 0.5) * 2
        selu = abs(roc_auc(Cu[:, i], (Yu == d).long()) - 0.5) * 2
        gaps.append(sel - selu)
    return {"bij_auc": bij_auc, "bij_nmi": bij_nmi, "named_best": named_best,
            "named_cover": named_cover, "mean_absR": mean_absR,
            "faith_gap": sum(gaps) / len(gaps)}


def summarize(name, accs, qs, usps=None, usps_cal=None):
    keys = ["bij_auc", "bij_nmi", "named_best", "mean_absR", "faith_gap"]

    def ms(v):
        t = torch.tensor(v); return t.mean().item(), t.std(unbiased=False).item()
    am, asd = ms(accs)
    out = f"{name:18} {am*100:5.2f}+/-{asd*100:.2f}"
    if usps is not None:
        um, usd = ms(usps)
        out += f"  {um*100:5.2f}+/-{usd*100:.2f}"
    if usps_cal is not None:
        cm, csd = ms(usps_cal)
        out += f"  {cm*100:5.2f}+/-{csd*100:.2f}"
    for k in keys:
        m, s = ms([q[k] for q in qs])
        out += f"  {m:5.2f}+/-{s:.2f}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--load-ontology", default=None,
                    help="evaluate a trained (hardened) OCBM checkpoint for the "
                         "ontology row instead of training a soft one")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ld, test_ld = make_loaders(args.data, args.batch_size)
    usps_ld = usps_loader(args.data, args.batch_size)
    spec = _bench.mnist_concept_spec("cpu")

    res = {h: {"acc": [], "usps": [], "usps_cal": [], "q": []}
           for h in ("ontology", "mlp", "linear")}
    if args.load_ontology:
        from concept_receptive_fields import load_model as _load_ocbm
        mo, _ = _load_ocbm(args.load_ontology, device)
        res["ontology"]["acc"].append(accuracy(mo, test_ld, device))
        res["ontology"]["usps"].append(accuracy(mo, usps_ld, device))
        res["ontology"]["usps_cal"].append(usps_cal_acc(mo, test_ld, usps_ld, device))
        res["ontology"]["q"].append(
            concept_quality(mo, test_ld, usps_ld, device, args.K, spec))
        print(f"loaded hardened OCBM {os.path.basename(args.load_ontology)}")
    for s in range(args.seeds):
        if not args.load_ontology:
            print(f"[seed {s}] training ontology / mlp / linear ...")
            acc_o, mo, nr = train_ontology_stable(args.K, train_ld, test_ld, device,
                                                  epochs=args.epochs, seed=s)
            if nr:
                print(f"  (ontology needed {nr} restart(s) to avoid collapse)")
            res["ontology"]["acc"].append(acc_o)
            res["ontology"]["usps"].append(accuracy(mo, usps_ld, device))
            res["ontology"]["usps_cal"].append(usps_cal_acc(mo, test_ld, usps_ld, device))
            res["ontology"]["q"].append(
                concept_quality(mo, test_ld, usps_ld, device, args.K, spec))
        for h in ("mlp", "linear"):
            m = train_blackbox(h, args.K, train_ld, test_ld, device, args.epochs, s)
            res[h]["acc"].append(accuracy(m, test_ld, device))
            res[h]["usps"].append(accuracy(m, usps_ld, device))
            res[h]["usps_cal"].append(usps_cal_acc(m, test_ld, usps_ld, device))
            res[h]["q"].append(
                concept_quality(m, test_ld, usps_ld, device, args.K, spec))

    print(f"\nHEAD ABLATION  K={args.K}  seeds={args.seeds}  (same encoder, task-only)")
    print(f"{'head':18} {'acc%':>11} {'USPSraw%':>11} {'USPScal%':>11}  {'bijAUC':>10} "
          f"{'bijNMI':>10} {'nameAUC':>10} {'mean|R|':>10} {'faithGap':>10}")
    print(summarize("ontology (OCBM)", res["ontology"]["acc"], res["ontology"]["q"],
                    res["ontology"]["usps"], res["ontology"]["usps_cal"]))
    print(summarize("black-box mlp", res["mlp"]["acc"], res["mlp"]["q"],
                    res["mlp"]["usps"], res["mlp"]["usps_cal"]))
    print(summarize("black-box linear", res["linear"]["acc"], res["linear"]["q"],
                    res["linear"]["usps"], res["linear"]["usps_cal"]))
    print("\nUSPSraw% = whole-model zero-shot accuracy on external USPS (no retrain, "
          "no calibration); USPScal% = after head-agnostic per-concept quantile "
          "calibration; bijAUC/bijNMI = strict one-to-one alignment to "
          "distinct human strokes; nameAUC = lenient best-match; lower mean|R| = distinct.")


if __name__ == "__main__":
    main()
