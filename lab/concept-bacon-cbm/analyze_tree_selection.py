r"""Per-class analysis: does the BINARY (left-fold) tree beat the FULL tree for
some CUB classes, and would per-class tree selection help?

Loads the two best supervised heads on the SAME CUB test set:
  * full tree  -- cub_ocbm_k112_fulltree_b8_eg2_neg_sup1_coef_800ep.pt (71.25%)
  * left tree  -- cub_ocbm_k112_sup_800ep.pt (67.28%, VectorTreeLogicHead)

Reports:
  1. overall accuracy of each (sanity),
  2. per-class recall; how many classes each head wins and by how much,
  3. ORACLE-UNION accuracy (either head correct) = ceiling of any routing,
  4. per-class-ROUTED predictor (route each class to its better head), raw and
     z-score-calibrated, with a held-out (select-on-half / eval-on-other) version
     so the routing gain is not just test-set selection overfitting.

NOTE: the two heads are SEPARATE models (different backbones/concepts), so this
is an ensemble-flavoured upper bound; a true per-class-tree model would share the
backbone. Runs on CPU by default so it does not disturb a running GPU job.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                    # noqa: E402
from cub_emergent import CUBEmergent                            # noqa: E402
from interpret_fulltree_cub import load_model, load_species_names  # noqa: E402

FULL = os.path.join(_HERE, "saved", "cub_ocbm_k112_fulltree_b8_eg2_neg_sup1_coef_800ep.pt")
LEFT = os.path.join(_HERE, "saved", "cub_ocbm_k112_sup_800ep.pt")


@torch.no_grad()
def run_model(model, loader, device):
    """Return truths [N,200], preds [N], labels [N]."""
    model.eval()
    ts, ys = [], []
    for img, c, y in loader:
        _, _, t = model(img.to(device))
        ts.append(t.cpu())
        ys.append(y)
    T = torch.cat(ts)
    y = torch.cat(ys)
    return T, T.argmax(1), y


def load_left(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    K = int(ck.get("K", 112))
    model = CUBEmergent(K, n_species=200, head="tree")
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if missing or unexpected:
        print(f"  [left load] missing={missing[:3]} unexpected={unexpected[:3]}")
    return model.to(device).eval()


def per_class_recall(pred, y, n=200):
    rec = torch.zeros(n)
    for c in range(n):
        m = y == c
        rec[c] = (pred[m] == c).float().mean() if m.any() else float("nan")
    return rec


def routed_acc(Tf, Tl, y, pick_full, calibrate=False):
    """Combined predictor: class c's score comes from full if pick_full[c] else left."""
    if calibrate:
        Tf = (Tf - Tf.mean(0)) / (Tf.std(0) + 1e-6)
        Tl = (Tl - Tl.mean(0)) / (Tl.std(0) + 1e-6)
    S = torch.where(pick_full.unsqueeze(0), Tf, Tl)     # [N,200]
    return (S.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=FULL)
    ap.add_argument("--left", default=LEFT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = torch.device(args.device)

    from torch.utils.data import DataLoader
    loader = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                        shuffle=False, num_workers=args.workers, pin_memory=False)
    names = load_species_names()

    print("loading models ...", flush=True)
    mfull, *_ = load_model(args.full, device)
    mleft = load_left(args.left, device)

    print("running full tree ...", flush=True)
    Tf, pf, y = run_model(mfull, loader, device)
    print("running left tree ...", flush=True)
    Tl, pl, y2 = run_model(mleft, loader, device)
    assert torch.equal(y, y2)

    accf = (pf == y).float().mean().item()
    accl = (pl == y).float().mean().item()
    print("\n===== overall =====")
    print(f"  full tree : {accf*100:.2f}%")
    print(f"  left tree : {accl*100:.2f}%")

    rf = per_class_recall(pf, y)
    rl = per_class_recall(pl, y)
    valid = ~(torch.isnan(rf) | torch.isnan(rl))
    left_wins = ((rl > rf) & valid)
    full_wins = ((rf > rl) & valid)
    ties = ((rf == rl) & valid)
    print("\n===== per-class recall =====")
    print(f"  classes where LEFT strictly better : {int(left_wins.sum())}")
    print(f"  classes where FULL strictly better : {int(full_wins.sum())}")
    print(f"  ties                               : {int(ties.sum())}")
    print(f"  mean margin when left wins : "
          f"{(rl - rf)[left_wins].mean().item()*100:.1f} pt")
    print(f"  mean margin when full wins : "
          f"{(rf - rl)[full_wins].mean().item()*100:.1f} pt")
    # biggest left-tree wins
    margin = (rl - rf)
    order = torch.argsort(torch.where(valid, margin, torch.full_like(margin, -9)),
                          descending=True)
    print("\n  top classes where BINARY(left) tree beats full tree:")
    for c in order[:12].tolist():
        if not bool(left_wins[c]):
            break
        print(f"    +{margin[c]*100:4.0f}pt  left {rl[c]*100:5.1f}  full {rf[c]*100:5.1f}"
              f"   {names[c]}")

    # oracle union (either head correct) = ceiling of any routing/ensemble
    union = ((pf == y) | (pl == y)).float().mean().item()
    print("\n===== routing upper bounds =====")
    print(f"  ORACLE per-IMAGE union (either correct): {union*100:.2f}%  "
          f"(ceiling; +{(union-accf)*100:.2f} over full)")

    # per-class routing (oracle selection on full test set)
    pick = rf >= rl                                     # True -> use full
    print(f"  per-class routed (oracle sel, raw)      : {routed_acc(Tf,Tl,y,pick)*100:.2f}%")
    print(f"  per-class routed (oracle sel, z-cal)    : {routed_acc(Tf,Tl,y,pick,True)*100:.2f}%")

    # full ENSEMBLE (both heads vote on all classes) -- the natural way to use
    # the diversity that the per-image oracle exposes.
    Tfz = (Tf - Tf.mean(0)) / (Tf.std(0) + 1e-6)
    Tlz = (Tl - Tl.mean(0)) / (Tl.std(0) + 1e-6)
    ens_z = ((Tfz + Tlz).argmax(1) == y).float().mean().item()
    ens_raw = ((Tf + Tl).argmax(1) == y).float().mean().item()
    print(f"  ENSEMBLE avg (z-cal, all classes)       : {ens_z*100:.2f}%  "
          f"(+{(ens_z-accf)*100:.2f} over full)")
    print(f"  ENSEMBLE avg (raw, all classes)         : {ens_raw*100:.2f}%")

    # ---- PER-IMAGE CONFIDENCE ROUTER (single-tree explainability) ----------
    # For each image pick the ONE tree that is more decisive (top1-top2 truth
    # margin), predict with that tree alone, and explain only that tree. This is
    # deployable (uses no labels) and preserves "explain one tree at a time",
    # unlike the blended ensemble and unlike the label-peeking oracle.
    def _margin(T):
        top2 = T.topk(2, dim=1).values                          # [N,2]
        return top2[:, 0] - top2[:, 1]                          # decisiveness

    print("\n===== per-image CONFIDENCE router (single-tree, deployable) =====")
    for tag, Ta, Tb in [("raw   ", Tf, Tl), ("z-cal ", Tfz, Tlz)]:
        cf, cl = _margin(Ta), _margin(Tb)
        use_full = cf >= cl                                     # per IMAGE
        pred = torch.where(use_full, Ta.argmax(1), Tb.argmax(1))
        acc = (pred == y).float().mean().item()
        frac_full = use_full.float().mean().item()
        # accuracy accounted purely by the routing decision (where trees disagree)
        disagree = Ta.argmax(1) != Tb.argmax(1)
        print(f"  route by {tag}margin : {acc*100:.2f}%  "
              f"(+{(acc-accf)*100:.2f} over full; picks full {frac_full*100:.0f}% "
              f"of images; trees disagree on {disagree.float().mean().item()*100:.0f}%)")

    # honest held-out temperature-free variant: the confidence router has no
    # fitted parameters, but z-cal uses test-set column stats (transductive).
    # Report a half/half honest z-cal number so the gain is not a stats leak.
    g0 = torch.Generator().manual_seed(0)
    perm0 = torch.randperm(y.size(0), generator=g0)
    A0, B0 = perm0[:y.size(0)//2], perm0[y.size(0)//2:]
    accs_r = []
    for sel, ev in [(A0, B0), (B0, A0)]:
        mf, sf = Tf[sel].mean(0), Tf[sel].std(0) + 1e-6
        ml, sl = Tl[sel].mean(0), Tl[sel].std(0) + 1e-6
        Sfz, Slz = (Tf[ev]-mf)/sf, (Tl[ev]-ml)/sl
        uf = _margin(Sfz) >= _margin(Slz)
        pr = torch.where(uf, Sfz.argmax(1), Slz.argmax(1))
        accs_r.append((pr == y[ev]).float().mean().item())
    print(f"  route by z-cal margin (HELD-OUT stats)  : {sum(accs_r)/2*100:.2f}%  "
          f"(honest; vs full {accf*100:.2f})")

    # honest held-out: select classes on half A, evaluate on half B (and swap)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(y.size(0), generator=g)
    A, Bx = perm[:y.size(0)//2], perm[y.size(0)//2:]
    accs = []
    for sel, ev in [(A, Bx), (Bx, A)]:
        rfa = per_class_recall(pf[sel], y[sel]); rla = per_class_recall(pl[sel], y[sel])
        pk = torch.where(torch.isnan(rfa) | torch.isnan(rla),
                         torch.ones_like(rfa, dtype=torch.bool), rfa >= rla)
        Sf, Sl, ye = Tf[ev], Tl[ev], y[ev]
        Sfz = (Sf - Sf.mean(0)) / (Sf.std(0)+1e-6)
        Slz = (Sl - Sl.mean(0)) / (Sl.std(0)+1e-6)
        S = torch.where(pk.unsqueeze(0), Sfz, Slz)
        accs.append((S.argmax(1) == ye).float().mean().item())
    print(f"  per-class routed (HELD-OUT sel, z-cal)  : {sum(accs)/2*100:.2f}%  "
          f"(honest; vs full {accf*100:.2f})")


if __name__ == "__main__":
    main()
