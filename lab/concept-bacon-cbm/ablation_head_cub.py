"""CUB version of the head ablation: graded-logic ONTOLOGY vs. black-box MLP.

Same ResNet-18 backbone + K-dim sigmoid concept bottleneck, species-label-only
supervision; swaps only the reasoning head. Reports species accuracy and
concept-alignment to the 112 human CUB attributes:

  * bijAUC  -- Hungarian one-to-one alignment of K concepts to K distinct attrs
  * bestAUC -- mean per-concept best-attribute AUC (lenient)
  * pos@.70 -- # concepts whose best attribute AUC >= 0.70
  * mean|R| -- distinctness

    python ablation_head_cub.py --K 8 --epochs 4 --seeds 1     # small POC
    python ablation_head_cub.py --K 24 --epochs 60 --seeds 2   # full
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                     # noqa: E402
from eval_shapes import roc_auc                                 # noqa: E402
from cub_emergent import (CUBEmergent, train_one, collect,      # noqa: E402
                          evaluate, mean_abs_corr)
from scipy.optimize import linear_sum_assignment               # noqa: E402


class CUBBlackBox(nn.Module):
    def __init__(self, K, n_species=200, head="mlp", hidden=256):
        super().__init__()
        self.backbone = _cub._make_resnet()
        self.concept = nn.Linear(512, K)
        self.head = (nn.Linear(K, n_species) if head == "linear" else
                     nn.Sequential(nn.Linear(K, hidden), nn.ReLU(),
                                   nn.Linear(hidden, n_species)))

    def concept_probs(self, x):
        return torch.sigmoid(self.concept(self.backbone(x)))

    def forward(self, x):
        c = self.concept_probs(x)
        return self.head(c), c, None


def train_blackbox(K, tl, vl, device, epochs, seed, head="mlp"):
    torch.manual_seed(seed)
    m = CUBBlackBox(K, head=head).to(device)
    bb_ids = {id(p) for p in m.backbone.parameters()}
    bb = [p for p in m.parameters() if id(p) in bb_ids]
    heads = [p for p in m.parameters() if id(p) not in bb_ids]
    opt = torch.optim.Adam([{"params": bb, "lr": 1e-4},
                            {"params": heads, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best, best_state = 0.0, None
    for ep in range(epochs):
        m.train()
        for img, c, y in tl:
            img, y = img.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(m(img)[0], y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sched.step()
        acc = evaluate(m, vl, device)
        if acc > best:
            best, best_state = acc, copy.deepcopy(m.state_dict())
        print(f"    [bb-{head}] epoch {ep+1}/{epochs} test {acc*100:.2f}%")
    if best_state is not None:
        m.load_state_dict(best_state)
    return best, m


def train_supervised_cbm(tl, vl, device, epochs, seed, lam=1.0):
    """Standard concept-supervised CBM: K=112 bottleneck (one unit per CUB
    attribute), trained with species CE + lam * concept BCE against the 112
    attribute annotations, then a linear concept->species head."""
    torch.manual_seed(seed)
    m = CUBBlackBox(112, head="linear").to(device)
    bb_ids = {id(p) for p in m.backbone.parameters()}
    bb = [p for p in m.parameters() if id(p) in bb_ids]
    heads = [p for p in m.parameters() if id(p) not in bb_ids]
    opt = torch.optim.Adam([{"params": bb, "lr": 1e-4},
                            {"params": heads, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best, best_state = 0.0, None
    for ep in range(epochs):
        m.train()
        for img, c, y in tl:
            img, c, y = img.to(device), c.to(device), y.to(device)
            opt.zero_grad()
            feat = m.backbone(img)
            clogit = m.concept(feat)                            # concept logits
            logits = m.head(torch.sigmoid(clogit))
            loss = (F.cross_entropy(logits, y)
                    + lam * F.binary_cross_entropy_with_logits(clogit, c))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sched.step()
        acc = evaluate(m, vl, device)
        if acc > best:
            best, best_state = acc, copy.deepcopy(m.state_dict())
        print(f"    [cbm-sup] epoch {ep+1}/{epochs} test {acc*100:.2f}%")
    if best_state is not None:
        m.load_state_dict(best_state)
    return best, m


@torch.no_grad()
def auc_matrix(model, loader, device, K=None):
    """K x 112 direction-agnostic concept<->attribute AUC matrix + valid attrs.
    K is inferred from the model's concept width (supports the K=112 CBM)."""
    C, A = collect(model, loader, device)                      # (N,K), (N,112)
    K = C.shape[1]
    M = torch.full((K, A.shape[1]), 0.5)
    valid = []
    for a in range(A.shape[1]):
        col = A[:, a]
        if 0 < col.sum() < len(col):
            valid.append(a)
            cl = col.long()
            for i in range(K):
                au = roc_auc(C[:, i], cl)
                M[i, a] = max(au, 1 - au)
    return M, valid, C


def alignment_from_M(M, C, fam=None):
    """Concept<->attribute quality from the K x 112 AUC matrix.

    Adds two anti-gaming metrics on top of the Hungarian matched alignment
    (``bij``): ``sel`` = mean per-concept (best - 2nd-best) attribute-AUC gap
    (a concept correlated with a whole family of attributes has low selectivity),
    and ``cov`` = number of distinct attribute FAMILIES the concepts cover when
    each is matched one-to-one and the match is confident (AUC >= 0.70)."""
    ri, ci = linear_sum_assignment(-M.numpy())
    best = M.amax(1)
    top2 = M.topk(2, dim=1).values
    sel = (top2[:, 0] - top2[:, 1]).mean().item()
    cov = 0
    if fam is not None:
        fams = {int(fam[ci[k]]) for k in range(len(ci))
                if M[ri[k], ci[k]] >= 0.70}
        cov = len(fams)
    return {"bij": M[ri, ci].mean().item(), "best": best.mean().item(),
            "pos": int((best >= 0.70).sum().item()), "corr": mean_abs_corr(C),
            "sel": sel, "cov": cov}


def faith_gap(M_cub, M_nab, common):
    """Per-concept best-attribute selectivity gap CUB->NABirds (MNIST faith_gap
    analog): selectivity = 2*(AUC-0.5); low gap = concept keeps its meaning on a
    different bird dataset. Restricted to attributes that vary in BOTH domains."""
    common = sorted(common)
    if not common:
        return float("nan")
    gaps = []
    for i in range(M_cub.shape[0]):
        a = max(common, key=lambda aa: M_cub[i, aa].item())
        gaps.append(2 * (M_cub[i, a].item() - 0.5)
                    - 2 * (M_nab[i, a].item() - 0.5))
    return sum(gaps) / len(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--load-ontology", default=None,
                    help="evaluate a trained (hardened) CUB OCBM checkpoint for "
                         "the ontology row instead of training a soft one")
    ap.add_argument("--nabirds", default=None,
                    help="path to NABirds root; adds a cross-dataset faithfulness "
                         "gap column (per-concept best-attr selectivity CUB->NABirds)")
    ap.add_argument("--save-heads", default=None,
                    help="directory to checkpoint the trained MLP/linear black-box "
                         "models (for the cross-model concept visualization)")
    ap.add_argument("--with-cbm", action="store_true",
                    help="also train a concept-SUPERVISED CBM (K=112, concept BCE "
                         "on the attributes) as a labeled reference row")
    ap.add_argument("--skip-blackbox", action="store_true",
                    help="skip the task-only MLP/linear rows (e.g. to add just the "
                         "supervised-CBM row to an existing table)")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="concept-BCE weight for the supervised CBM")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl = DataLoader(_cub._CUBImages("train", True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)

    na_loader = None
    if args.nabirds:
        import nabirds
        na_loader = DataLoader(nabirds.NABirdsCUB(args.nabirds), batch_size=128,
                               shuffle=False, num_workers=args.workers,
                               pin_memory=True)

    # attribute families (has_wing_color, has_bill_shape, ...) for coverage
    _n112, fam, type_names, _ = _cub._load_attr_groups()
    n_families = len(type_names)

    def quality(model):
        M_cub, valid_cub, C_cub = auc_matrix(model, vl, device)
        q = alignment_from_M(M_cub, C_cub, fam)
        if na_loader is not None:
            M_nab, valid_nab, _ = auc_matrix(model, na_loader, device)
            q["faith"] = faith_gap(M_cub, M_nab,
                                   set(valid_cub) & set(valid_nab))
            # whole-model zero-shot species accuracy on the mapped NABirds images
            # (validity check: is the model even functional on NABirds?)
            q["na_acc"] = evaluate(model, na_loader, device)
        return q

    res = {"ontology": {"acc": [], "q": []}, "mlp": {"acc": [], "q": []},
           "linear": {"acc": [], "q": []}, "cbm": {"acc": [], "q": []}}
    if args.load_ontology:
        mo = CUBEmergent(args.K, head="tree").to(device)
        ckpt = torch.load(args.load_ontology, map_location=device)
        mo.load_state_dict(ckpt["state_dict"], strict=False)
        mo.eval()
        res["ontology"]["acc"].append(evaluate(mo, vl, device))
        res["ontology"]["q"].append(quality(mo))
        print(f"loaded hardened CUB OCBM {os.path.basename(args.load_ontology)}")
    for s in range(args.seeds):
        if not args.load_ontology:
            print(f"[seed {s}] ontology (tree) ...")
            acc_o, mo = train_one(args.K, tl, vl, device, epochs=args.epochs, seed=s)
            res["ontology"]["acc"].append(acc_o)
            res["ontology"]["q"].append(quality(mo))
        if not args.skip_blackbox:
            for h in ("mlp", "linear"):
                print(f"[seed {s}] black-box {h} ...")
                acc_b, mb = train_blackbox(args.K, tl, vl, device, args.epochs, s, head=h)
                res[h]["acc"].append(acc_b)
                res[h]["q"].append(quality(mb))
                if args.save_heads and s == 0:
                    os.makedirs(args.save_heads, exist_ok=True)
                    path = os.path.join(args.save_heads, f"cub_bb_{h}.pt")
                    torch.save({"state_dict": mb.state_dict(), "K": args.K,
                                "head": h}, path)
                    print(f"    saved {path}")
        if args.with_cbm:
            print(f"[seed {s}] supervised CBM (K=112, concept loss, lam={args.lam}) ...")
            acc_c, mc = train_supervised_cbm(tl, vl, device, args.epochs, s, lam=args.lam)
            res["cbm"]["acc"].append(acc_c)
            res["cbm"]["q"].append(quality(mc))
            if args.save_heads and s == 0:
                os.makedirs(args.save_heads, exist_ok=True)
                path = os.path.join(args.save_heads, "cub_cbm_sup.pt")
                torch.save({"state_dict": mc.state_dict(), "K": 112,
                            "head": "linear"}, path)
                print(f"    saved {path}")

    def ms(v):
        t = torch.tensor([float(x) for x in v])
        return t.mean().item(), t.std(unbiased=False).item()

    print(f"\nCUB HEAD ABLATION  K={args.K}  seeds={args.seeds}")
    hdr = (f"{'head':16} {'acc%':>11} {'bijAUC':>10} {'bestAUC':>10} "
           f"{'pos@.70':>8} {'mean|R|':>9} {'select':>8} {'cover':>7}")
    if na_loader is not None:
        hdr += f" {'faithGap':>9} {'NA-acc%':>7}"
    print(hdr)
    labels = {"ontology": "ontology", "mlp": "mlp", "linear": "linear",
              "cbm": "cbm(K=112,sup)"}
    for name in ("ontology", "mlp", "linear", "cbm"):
        if not res[name]["acc"]:
            continue
        am, asd = ms(res[name]["acc"])
        bij = ms([q["bij"] for q in res[name]["q"]])
        best = ms([q["best"] for q in res[name]["q"]])
        pos = ms([q["pos"] for q in res[name]["q"]])
        corr = ms([q["corr"] for q in res[name]["q"]])
        sel = ms([q["sel"] for q in res[name]["q"]])
        cov = ms([q["cov"] for q in res[name]["q"]])
        row = (f"{labels[name]:16} {am*100:5.2f}+/-{asd*100:.2f} {bij[0]:5.2f}+/-{bij[1]:.2f} "
               f"{best[0]:5.2f}+/-{best[1]:.2f} {pos[0]:5.1f} {corr[0]:6.2f} "
               f"{sel[0]:7.2f} {cov[0]:6.1f}")
        if na_loader is not None:
            fg = ms([q["faith"] for q in res[name]["q"]])
            na = ms([q["na_acc"] for q in res[name]["q"]])
            row += f" {fg[0]:8.2f} {na[0]*100:7.2f}"
        print(row)
    print("\nbijAUC = strict 1-to-1 concept<->attribute alignment; pos@.70 = "
          "# concepts matching a human attribute.")
    print(f"select = mean per-concept (best - 2nd-best) attribute-AUC gap "
          f"(higher = more identifiable, harder to game with redundant concepts); "
          f"cover = # distinct attribute families (of {n_families}) covered by the "
          f"one-to-one match at AUC>=0.70 (higher = broader ontology).")
    if na_loader is not None:
        print("faithGap = mean per-concept best-attribute selectivity drop "
              "CUB->NABirds (lower = concepts keep meaning cross-dataset); "
              "NA-acc% = whole-model zero-shot species accuracy on the mapped "
              "NABirds images (validity check; chance ~0.5%).")


if __name__ == "__main__":
    main()
