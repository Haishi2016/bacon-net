"""
Generate 200 manually-constructed BACON trees for CUB-200-2011 and measure their
discriminability ceiling -- WITHOUT any training.

Each bird species gets one fixed tree built from its concept *prototype* (the
majority-vote of the 112 binary attributes over that class's training images).
Two tree designs are compared:

  * signed-AND : class_k = AND over ALL 112 concepts, each as a positive literal
                 if it is ON in the prototype, else a NOT literal.  (Exact
                 graded template match.)
  * positive-AND: class_k = AND over only the concepts that are ON in the
                 prototype.  (Sparser / more readable, but prone to the "subset"
                 problem where a species whose signature is contained in another's
                 also fires.)

The "ceiling" is the task accuracy when the *true* concept labels are fed into
the fixed trees (argmax over the 200 tree truths) -- i.e. how well the manually
generated trees separate the 200 classes given perfect concepts.  No CNN, no
training; this validates the tree set before we train a concept extractor.

    python cub_trees.py
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch

CUB = r"C:\School\datasets\cub\CUB_200_2011"
_EPS = 1e-6


def load_split(name: str):
    with open(os.path.join(CUB, f"{name}.pkl"), "rb") as f:
        data = pickle.load(f)
    C = torch.tensor([d["attribute_label"] for d in data], dtype=torch.float32)  # (N,112)
    y = torch.tensor([d["class_label"] for d in data], dtype=torch.long)          # (N,)
    return C, y


def build_prototypes(C: torch.Tensor, y: torch.Tensor, n_classes=200) -> torch.Tensor:
    """Per-class majority-vote concept signature -> P (n_classes, 112) in {0,1}."""
    K = C.shape[1]
    P = torch.zeros(n_classes, K)
    for k in range(n_classes):
        m = y == k
        if m.any():
            P[k] = (C[m].mean(0) >= 0.5).float()
    return P


def load_frequencies(P: torch.Tensor):
    """Per-class attribute FREQUENCY aligned to the 112 concepts -> F (200,112) in [0,1].

    CUB ships class_attribute_labels_continuous.txt (200x312, % of class images
    with each attribute).  We align its 312 columns to the pkl's 112 concepts by
    matching each binary prototype column to the continuous column that best
    reproduces it across the 200 classes.
    """
    path = os.path.join(CUB, "attributes", "class_attribute_labels_continuous.txt")
    cont = torch.tensor(np.loadtxt(path), dtype=torch.float32) / 100.0   # (200,312)
    cb = (cont >= 0.5).float()
    idx, exact = [], 0
    for j in range(P.shape[1]):
        d = (cb != P[:, j:j + 1]).float().sum(0)                         # (312,)
        j_best = int(d.argmin())
        idx.append(j_best)
        exact += int(d[j_best].item() == 0)
    F = cont[:, idx].clone()
    return F, exact



def graded_and(match: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Graded AND (geometric mean) over the last dim, optionally masked.

    match: (..., K) agreement values in [0,1].  mask: (..., K) 1 where the literal
    participates.  Returns (...,) truth in [0,1].
    """
    logm = torch.log(match.clamp_min(_EPS))
    if mask is None:
        return torch.exp(logm.mean(-1))
    denom = mask.sum(-1).clamp_min(1.0)
    return torch.exp((logm * mask).sum(-1) / denom)


def tree_truths(C: torch.Tensor, P: torch.Tensor, design: str) -> torch.Tensor:
    """C: (N,112) concept probs. P: (200,112). Returns (N,200) per-class truth."""
    N, K = C.shape
    out = torch.empty(N, P.shape[0])
    for i in range(0, N, 512):                      # chunk over samples
        c = C[i:i + 512].unsqueeze(1)               # (b,1,K)
        p = P.unsqueeze(0)                          # (1,200,K)
        if design == "signed":
            match = p * c + (1 - p) * (1 - c)        # agreement per concept
            out[i:i + 512] = graded_and(match)
        else:                                        # positive-only AND
            match = c.expand(-1, P.shape[0], -1)     # (b,200,K)
            out[i:i + 512] = graded_and(match, mask=p.expand(c.shape[0], -1, -1))
    return out


def weighted_truths(C: torch.Tensor, F: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Statistical weighted graded-AND tree.

    For each class the majority DIRECTION of a concept (positive literal if
    freq>=0.5, else NOT) is weighted by its CONFIDENCE ``|2*freq-1|**gamma`` -- so
    an always-on / never-on feature is a strong (near-mandatory) literal, while a
    ~50% feature is nearly optional (weight ~0) and barely affects the truth.
    C: (N,112) -> (N,200).
    """
    sign = (F >= 0.5).float()                       # (200,K) literal direction
    w = (2.0 * F - 1.0).abs().pow(gamma)            # (200,K) confidence weight
    N = C.shape[0]
    out = torch.empty(N, F.shape[0])
    for i in range(0, N, 512):
        c = C[i:i + 512].unsqueeze(1)               # (b,1,K)
        s = sign.unsqueeze(0)                       # (1,200,K)
        match = s * c + (1 - s) * (1 - c)            # (b,200,K)
        logm = torch.log(match.clamp_min(_EPS))
        out[i:i + 512] = torch.exp((logm * w).sum(-1) / w.sum(-1).clamp_min(_EPS))
    return out


def ceiling(C, y, P, design):
    truths = tree_truths(C, P, design)
    pred = truths.argmax(1)
    return (pred == y).float().mean().item()


def main():
    Ctr, ytr = load_split("train")
    Cte, yte = load_split("test")
    print(f"train {tuple(Ctr.shape)}  test {tuple(Cte.shape)}  "
          f"classes {int(ytr.max()) + 1}  concepts {Ctr.shape[1]}")

    P = build_prototypes(Ctr, ytr)
    n_unique = len({tuple(row.tolist()) for row in P})
    print(f"\n200 prototypes -> {n_unique} distinct concept signatures")
    sizes = P.sum(1)
    print(f"concepts ON per class: mean {sizes.mean():.1f} (min {int(sizes.min())}, "
          f"max {int(sizes.max())})")

    print("\nManually-generated tree discriminability ceiling "
          "(true concepts -> argmax tree):")
    print(f"{'tree design':16s} {'train acc':>10s} {'test acc':>10s}")
    for design in ("signed", "positive"):
        a_tr = ceiling(Ctr, ytr, P, design)
        a_te = ceiling(Cte, yte, P, design)
        print(f"{design + '-AND':16s} {a_tr*100:9.2f} {a_te*100:9.2f}")

    # statistical (frequency-weighted) design
    F, exact = load_frequencies(P)
    print(f"\naligned {exact}/112 concepts to continuous frequencies exactly")
    for gamma in (0.5, 1.0, 2.0):
        a_tr = (weighted_truths(Ctr, F, gamma).argmax(1) == ytr).float().mean()
        a_te = (weighted_truths(Cte, F, gamma).argmax(1) == yte).float().mean()
        print(f"{'weighted g=' + str(gamma):16s} {a_tr*100:9.2f} {a_te*100:9.2f}")

    # example class: strong-positive / strong-NOT / optional features
    k = 0
    strong_pos = (F[k] > 0.9).sum().item()
    strong_neg = (F[k] < 0.1).sum().item()
    optional = ((F[k] > 0.35) & (F[k] < 0.65)).sum().item()
    print(f"\nexample class {k}: {strong_pos} strong-positive, {strong_neg} strong-NOT, "
          f"{optional} optional (~0.5) concepts (of 112)")

    # prototype self-consistency: feed the prototypes themselves
    Pself = P.clone()
    yself = torch.arange(P.shape[0])
    print(f"\nsigned-AND on the prototypes themselves (upper bound): "
          f"{ceiling(Pself, yself, P, 'signed')*100:.2f}%")


if __name__ == "__main__":
    main()
