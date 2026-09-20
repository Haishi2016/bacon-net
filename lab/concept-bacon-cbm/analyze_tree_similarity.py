"""Read-only structural analysis of the 200 per-class graded-logic trees in a
hardened fulltree OCBM, plus (optionally) whether structural similarity
CORRELATES WITH CLASS CONFUSION.

Question being tested: for hard/confusable classes, are the competing class-trees
STRUCTURALLY SIMILAR (same concepts / same logic)? If similarity tracks confusion,
a targeted discriminative regularizer might help; if not, redundancy is benign.

Two parts:
  * STRUCTURE (CPU, instant): reads the frozen DAG buffers -> per-class
    concept-usage / signed-usage / root-andness signatures -> 200x200 similarity.
  * CONFUSION (--with-confusion, one forward pass over the test set): confusion
    matrix + Spearman(confusion, tree-similarity) + hard-class partner analysis.

Nothing here trains or mutates the checkpoint.
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                    # noqa: E402
from cub_emergent import CUBEmergent                           # noqa: E402


# ----------------------------------------------------------------- load
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CUBEmergent(ck["K"], head=ck["head"], branching=ck["branching"],
                        fulltree_negation=ck.get("negation", False),
                        fulltree_max_egress=ck.get("max_egress", 1),
                        backbone_kind=ck["backbone"]).to(device)
    model.head.freeze_egress()                                 # allocate frozen buffers + hard path
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()
    return model, ck, len(missing), len(unexpected)


def class_names():
    for cand in (os.path.join(_cub.CUB, "classes.txt"),
                 os.path.join(_cub.CUB, "CUB_200_2011", "classes.txt")):
        if os.path.exists(cand):
            names = {}
            with open(cand, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.split(None, 1)
                    if len(p) == 2:
                        names[int(p[0]) - 1] = p[1].strip().split(".", 1)[-1]
            return [names.get(i, f"class{i}") for i in range(200)]
    return [f"class{i}" for i in range(200)]


# ----------------------------------------------------------------- structure
@torch.no_grad()
def leaf_influence(head):
    """Per-class [H, K] influence of each input concept on the root, from the
    frozen DAG: child weights (frozen edges normalized over sources, orphan->
    uniform) propagated root->leaves. Mirrors _child_weights, so this is the
    exact structural contribution of each concept to the class truth."""
    H = head.num_heads
    infl = torch.ones(H, 1, device=head.frozen_route_0.device)          # root
    for l in reversed(range(head.depth)):
        E = getattr(head, f"frozen_route_{l}")                          # [H, w_in, w_out]
        col = E.sum(dim=1, keepdim=True)
        w = E / col.clamp_min(1e-8)
        dead = (col < 1e-6).expand_as(w)
        if bool(dead.any()):
            w = torch.where(dead, E.new_full(w.shape, 1.0 / E.size(1)), w)
        infl = torch.einsum("hsd,hd->hs", w, infl)                     # [H, w_in]
    return infl                                                        # [H, K]


@torch.no_grad()
def structure_signatures(head):
    K = head.input_size
    infl = leaf_influence(head)                                        # [H, K]
    usage = infl / infl.sum(dim=1, keepdim=True).clamp_min(1e-9)       # rows sum to 1
    if head.use_negation and bool(head.transform_frozen):
        sign = head.frozen_transform * 2.0 - 1.0                       # 1=identity,0=NOT -> +/-1
    else:
        sign = torch.ones_like(usage)
    signed = usage * sign
    # root andness (final AND/OR-ness of each class rule): andness_bias last layer
    root_a = (torch.sigmoid(head.andness_bias[head.depth - 1]) * 3.0 - 1.0).squeeze(-1)  # [H]
    # effective concept count = # concepts covering 90% of a class's influence mass
    srt, _ = usage.sort(dim=1, descending=True)
    cum = srt.cumsum(dim=1)
    n_eff = (cum < 0.90).sum(dim=1) + 1                                # [H]
    return usage.cpu().numpy(), signed.cpu().numpy(), root_a.cpu().numpy(), n_eff.cpu().numpy()


def cosine_matrix(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def report_structure(usage, signed, root_a, n_eff, names):
    H = usage.shape[0]
    S = cosine_matrix(usage)                                           # unsigned concept-usage sim
    Ss = cosine_matrix(signed)                                         # signed (polarity-aware)
    iu = np.triu_indices(H, k=1)
    su, ss = S[iu], Ss[iu]
    print("===== STRUCTURE (200 per-class trees) =====", flush=True)
    print(f"effective concepts/class: mean {n_eff.mean():.1f}  median {int(np.median(n_eff))}  "
          f"range {n_eff.min()}-{n_eff.max()}", flush=True)
    print(f"root andness: mean {root_a.mean():.2f}  (>=1.5 hard-AND, ~1 AND, ~0.5 mean, ~0 OR)", flush=True)
    print(f"pairwise concept-usage cosine: mean {su.mean():.3f}  median {np.median(su):.3f}  "
          f"90th pct {np.percentile(su, 90):.3f}  max {su.max():.3f}", flush=True)
    print(f"  pairs >0.90 sim: {(su > 0.90).sum()} / {len(su)}  ({100 * (su > 0.90).mean():.2f}%)  "
          f">0.80: {(su > 0.80).sum()}", flush=True)
    print(f"pairwise SIGNED (polarity-aware) cosine: mean {ss.mean():.3f}  "
          f"(gap to unsigned = negation flips)", flush=True)
    # top most-similar different-class pairs
    order = np.argsort(-S[iu])
    print("\n  top-12 most structurally similar DIFFERENT-class tree pairs:", flush=True)
    for r in order[:12]:
        i, j = iu[0][r], iu[1][r]
        print(f"    {S[i, j]:.3f}  [{i:3d}] {names[i][:26]:26s}  ~  [{j:3d}] {names[j][:26]}", flush=True)
    return S


# ----------------------------------------------------------------- confusion
@torch.no_grad()
def predict(model, loader, device):
    preds, ys = [], []
    for img, c, y in loader:
        logits, _, _ = model(img.to(device))
        preds.append(logits.argmax(1).cpu())
        ys.append(y)
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def report_confusion(S, model, device, names):
    image_size = model.backbone_size
    from torch.utils.data import DataLoader
    vl = DataLoader(_cub._CUBImages("test", False, attr312=True, image_size=image_size),
                    batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
    preds, ys = predict(model, vl, device)
    H = 200
    C = np.zeros((H, H), dtype=np.int64)
    for t, p in zip(ys, preds):
        C[t, p] += 1
    acc = (preds == ys).mean()
    recall = np.array([C[i, i] / max(C[i].sum(), 1) for i in range(H)])
    conf = C + C.T
    np.fill_diagonal(conf, 0)
    print(f"\n===== CONFUSION (test acc {acc * 100:.2f}%) =====", flush=True)

    iu = np.triu_indices(H, k=1)
    x = conf[iu].astype(float)          # symmetric confusion count per pair
    y = S[iu]                           # tree similarity per pair
    mask = x > 0                        # only pairs that are ever confused
    try:
        from scipy.stats import spearmanr
        rho_all, _ = spearmanr(x, y)
        rho_conf, _ = spearmanr(x[mask], y[mask])
    except Exception:
        def _sp(a, b):
            ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
            return np.corrcoef(ra, rb)[0, 1]
        rho_all, rho_conf = _sp(x, y), _sp(x[mask], y[mask])
    print(f"Spearman(confusion, tree-similarity): all pairs {rho_all:+.3f}  |  "
          f"confused-only pairs {rho_conf:+.3f}  (n_confused={mask.sum()})", flush=True)
    print(f"mean tree-sim of CONFUSED pairs {y[mask].mean():.3f}  vs ALL pairs {y.mean():.3f}  "
          f"vs NON-confused {y[~mask].mean():.3f}", flush=True)

    hard = np.argsort(recall)[:10]
    print("\n  10 hardest classes (lowest recall): partner tree-sim vs random baseline:", flush=True)
    rng = np.random.default_rng(0)
    for i in hard:
        j = int(np.argmax(conf[i]))                     # top confusion partner
        rand_sim = float(S[i, rng.integers(0, H, 50)].mean())
        print(f"    [{i:3d}] {names[i][:24]:24s} recall {recall[i] * 100:4.1f}%  "
              f"-> confused w/ [{j:3d}] {names[j][:22]:22s} sim {S[i, j]:.3f}  "
              f"(rand {rand_sim:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        _HERE, "saved", "cub_ocbm_k312_fulltree_b8_eg2_neg_sup0p03_iw_inception_v3_150ep.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--structure-only", action="store_true",
                    help="skip the forward pass / confusion part (CPU, no GPU contention)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.structure_only:
        device = torch.device("cpu")
    model, ck, miss, unexp = load_model(args.ckpt, device)
    print(f"loaded {os.path.basename(args.ckpt)}  (K={ck['K']} egress={ck.get('max_egress')} "
          f"neg={ck.get('negation')}; load {miss} missing/{unexp} unexpected)", flush=True)

    names = class_names()
    usage, signed, root_a, n_eff = structure_signatures(model.head)
    S = report_structure(usage, signed, root_a, n_eff, names)

    if not args.structure_only:
        report_confusion(S, model, device, names)


if __name__ == "__main__":
    main()
