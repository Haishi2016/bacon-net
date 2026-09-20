"""Confusion-weighted (hard-negative) cross-learning loss -- idea A.

The current hybrid loss uses `balanced_ovr_loss`, which spreads each sample's
cross-class (negative) signal UNIFORMLY over the ~199 rival trees (weight
1/(C-1) each). But the confusion analysis showed cross-class error is
CONCENTRATED on a few lookalike rivals -- so ~99% of that budget is spent on
trivially-separable classes.

`confusion_weighted_ovr_loss` reallocates the negative budget to the rivals that
actually COMPETE on this sample: negative weight for tree j (j != y) is
proportional to how strongly tree j fires, ``truths_j ** hardness`` (detached),
renormalized per sample to sum to 1 -- so the overall scale matches
`balanced_ovr_loss` (drops into the hybrid with the same ``ovr_weight``).

  hardness = 0  -> uniform 1/(C-1)  == balanced_ovr_loss (exact).
  hardness = 1  -> linear hard-negative emphasis (competing rivals get pushed).
  hardness = 2  -> focal-like (only the strongest rivals get real signal).

This is "cross-learning" made TARGETED: a class-A image teaches specifically the
rival trees that mistake it for their class -- turning each rule toward "what
separates me from my nearest lookalike" (field-guide-style diagnostics).
"""
import torch
import torch.nn.functional as F


def confusion_weighted_ovr_loss(truths, y, n_classes, hardness=1.0,
                                conf_floor=None, eps=1e-8):
    """Hard-negative one-vs-rest loss. See module docstring.

    truths: (B, C) tree outputs in (0,1); y: (B,) class indices.
    hardness>=0: 0 == uniform balanced OvR; higher => concentrate on rivals that fire.
    conf_floor: optional confidence gate (floor + (1-floor)*truths[y], detached).
    """
    onehot = F.one_hot(y, n_classes).float()
    bce = F.binary_cross_entropy(truths, onehot, reduction="none")   # (B, C)
    pos = (bce * onehot).sum(1)                                      # weight 1
    neg_mask = 1.0 - onehot
    # rival "competition" weight: how strongly each wrong tree fires (detached ->
    # it is a WEIGHT, never a gradient path that could lower a rival for free).
    w = (truths.detach().clamp_min(eps) ** float(hardness)) * neg_mask
    w = w / (w.sum(1, keepdim=True) + eps)                           # per-sample sums to 1
    neg = (bce * w).sum(1)                                           # weighted-mean negative, O(1)
    if conf_floor is not None:
        pt = truths.gather(1, y.view(-1, 1)).squeeze(1).detach()
        gate = float(conf_floor) + (1.0 - float(conf_floor)) * pt
        neg = gate * neg
    return (pos + neg).mean()


# --------------------------------------------------------------------- self-test
def _balanced_ref(truths, y, C):
    onehot = F.one_hot(y, C).float()
    bce = F.binary_cross_entropy(truths, onehot, reduction="none")
    pos = (bce * onehot).sum(1)
    neg = (bce * (1.0 - onehot)).sum(1) / (C - 1)
    return (pos + neg).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 16, 20
    t = torch.rand(B, C).clamp(1e-4, 1 - 1e-4)
    y = torch.randint(0, C, (B,))

    # (1) hardness=0 must equal the uniform balanced OvR exactly.
    a = confusion_weighted_ovr_loss(t, y, C, hardness=0.0)
    b = _balanced_ref(t, y, C)
    print(f"hardness0 == balanced: {float(a):.6f} vs {float(b):.6f}  "
          f"diff {abs(float(a) - float(b)):.2e}")
    assert abs(float(a) - float(b)) < 1e-6

    # (2) perfect prediction -> ~0 loss.
    perfect = F.one_hot(y, C).float().clamp(1e-6, 1 - 1e-6)
    print(f"perfect-pred loss: {float(confusion_weighted_ovr_loss(perfect, y, C, 1.0)):.4f} (expect ~0)")

    # (3) a HARD rival (one wrong tree fires high) should receive MORE negative
    #     gradient under hardness=1 than under uniform (hardness=0).
    tt = torch.full((1, C), 0.05)
    tt[0, 0] = 0.95          # true class
    rival = 7
    tt[0, rival] = 0.9       # a wrong tree firing high = hard negative
    yy = torch.tensor([0])
    for h in (0.0, 1.0, 2.0):
        x = tt.clone().requires_grad_(True)
        confusion_weighted_ovr_loss(x, yy, C, hardness=h).backward()
        gr = x.grad[0, rival].item()
        gu = x.grad[0, 1].item()      # a non-firing rival (0.05)
        print(f"hardness {h}: grad on HARD rival {gr:+.4f}  vs quiet rival {gu:+.4f}  "
              f"(ratio {gr / (gu + 1e-9):.1f}x)")

    # (4) finite grads on a random batch.
    x = t.clone().requires_grad_(True)
    confusion_weighted_ovr_loss(x, y, C, hardness=1.0, conf_floor=0.2).backward()
    print(f"random-batch grad finite: {bool(torch.isfinite(x.grad).all())}")
    print("OK")
