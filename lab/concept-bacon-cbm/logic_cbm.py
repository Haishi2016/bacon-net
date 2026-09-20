r"""LogicCBM baseline (Vemuri et al., WACV 2026) reimplemented on our CUB concept
pipeline for a CONTROLLED interpret-reconstruct comparison.

Faithful re-implementation of the paper's CUB "Logic" head (their difflogic layer
+ weighted sum), in pure PyTorch (no difflogic CUDA kernel):

    backbone -> K concept sigmoids -> LogicLayer(K -> n_neurons) -> Linear(n_neurons -> C)

  * LogicLayer: ``n_neurons`` logic neurons, each wired to a RANDOM PAIR of
    concepts (difflogic ``connections='random'``) and applying one of the 16
    boolean gates via ``bin_op`` (their functional.py). With ``fixed_gates`` the
    gate per neuron is a FIXED random one-hot (requires_grad=False) -- matching
    the paper's ``-fixed_gates`` CUB command -- so only the final Linear (the
    weighted sum) learns the per-class combination. Without it, each neuron
    learns a softmax over the 16 gates (soft during train, argmax at eval).
  * Backbone + concept layer are IDENTICAL to our CUBEmergent so the comparison
    isolates the reasoning HEAD (LogicCBM vs our GL tree) on the SAME concepts.

Per-class rule = the 250 predicates ``gate_i(concept_a_i, concept_b_i)`` weighted
by the Linear row for that class; the interpretable rule = top-|weight| predicates.

    py -3 logic_cbm.py train --epochs 60 --concept-lam 1.0
    py -3 logic_cbm.py eval  --ckpt saved/logiccbm_k112_250n.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                          # noqa: E402


# ---- 16 differentiable boolean gates (difflogic/functional.py, bin_op) -------
def bin_op(a, b, i):
    if i == 0:  return torch.zeros_like(a)
    if i == 1:  return a * b                       # A AND B
    if i == 2:  return a - a * b                   # A AND NOT B  = NOT(A=>B)
    if i == 3:  return a                           # A
    if i == 4:  return b - a * b                   # NOT A AND B  = NOT(B=>A)
    if i == 5:  return b                           # B
    if i == 6:  return a + b - 2 * a * b           # A XOR B
    if i == 7:  return a + b - a * b               # A OR B
    if i == 8:  return 1 - (a + b - a * b)         # NOR
    if i == 9:  return 1 - (a + b - 2 * a * b)     # XNOR
    if i == 10: return 1 - b                        # NOT B
    if i == 11: return 1 - b + a * b               # B => A
    if i == 12: return 1 - a                        # NOT A
    if i == 13: return 1 - a + a * b               # A => B
    if i == 14: return 1 - a * b                    # NAND
    if i == 15: return torch.ones_like(a)


GATE_STR = {
    0: "F", 1: "{a} AND {b}", 2: "{a} AND NOT {b}", 3: "{a}", 4: "NOT {a} AND {b}",
    5: "{b}", 6: "{a} XOR {b}", 7: "{a} OR {b}", 8: "NOT({a} OR {b})",
    9: "{a} XNOR {b}", 10: "NOT {b}", 11: "{b} IMPLIES {a}", 12: "NOT {a}",
    13: "{a} IMPLIES {b}", 14: "NOT({a} AND {b})", 15: "T",
}


class LogicLayer(nn.Module):
    """difflogic layer: n_neurons logic gates over random concept pairs."""

    def __init__(self, in_dim, out_dim, fixed_gates=True, seed=0):
        super().__init__()
        self.in_dim, self.out_dim, self.fixed_gates = in_dim, out_dim, fixed_gates
        g = torch.Generator().manual_seed(seed)
        # difflogic 'random' connections: two random index vectors over inputs.
        c = torch.randperm(2 * out_dim, generator=g) % in_dim
        c = torch.randperm(in_dim, generator=g)[c].reshape(2, out_dim)
        self.register_buffer("a_idx", c[0].long())
        self.register_buffer("b_idx", c[1].long())
        if fixed_gates:
            gate_ids = torch.randint(0, 16, (out_dim,), generator=g)
            self.register_buffer("gate_ids", gate_ids)          # fixed random gate/neuron
            self.gate_logits = None
        else:
            self.gate_logits = nn.Parameter(torch.randn(out_dim, 16) * 0.1)
            self.register_buffer("gate_ids", torch.zeros(out_dim, dtype=torch.long))

    def gate_weights(self):
        """(out_dim, 16) gate mixture: one-hot (fixed / eval) or softmax (train)."""
        if self.fixed_gates:
            return F.one_hot(self.gate_ids, 16).float()
        if self.training:
            return F.softmax(self.gate_logits, dim=-1)
        return F.one_hot(self.gate_logits.argmax(-1), 16).float()

    def forward(self, x):                                       # x: (B, in_dim)
        a, b = x[:, self.a_idx], x[:, self.b_idx]               # (B, out_dim)
        w = self.gate_weights()                                 # (out_dim, 16)
        out = x.new_zeros(x.size(0), self.out_dim)
        for i in range(16):
            out = out + w[:, i] * bin_op(a, b, i)
        return out

    @torch.no_grad()
    def hard_gate_ids(self):
        return self.gate_ids if self.fixed_gates else self.gate_logits.argmax(-1)


class CUBLogicCBM(nn.Module):
    """backbone -> K concepts -> difflogic LogicLayer -> Linear -> class logits."""

    def __init__(self, K=112, n_classes=200, n_neurons=250, fixed_gates=True,
                 concept_scale=1.0, seed=0, backbone_kind="resnet18"):
        super().__init__()
        self.K = K
        self.concept_scale = float(concept_scale)
        self.backbone, feat_dim, self.backbone_size = _cub.make_backbone(backbone_kind)
        self.concept = nn.Linear(feat_dim, K)
        self.logic = LogicLayer(K, n_neurons, fixed_gates=fixed_gates, seed=seed)
        self.classifier = nn.Linear(n_neurons, n_classes)

    def concept_probs(self, x):
        return torch.sigmoid(self.concept_scale * self.concept(self.backbone(x)))

    def forward(self, x):
        c = self.concept_probs(x)
        logits = self.classifier(self.logic(c))
        return logits, c


# ------------------------------------------------------------------- train/eval
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for img, _c, y in loader:
        logits, _ = model(img.to(device))
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total += y.size(0)
    return correct / total


def train(args):
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    attr312 = bool(args.attr312)
    K = 312 if attr312 else args.K
    image_size = 299 if args.backbone == "inception_v3" else 224
    eval_bs = 64 if args.backbone == "inception_v3" else 128
    tl = DataLoader(_cub._CUBImages("train", True, attr312=attr312, image_size=image_size),
                    batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False, attr312=attr312, image_size=image_size),
                    batch_size=eval_bs,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    model = CUBLogicCBM(K=K, n_neurons=args.n_neurons,
                        fixed_gates=not args.learn_gates, seed=args.seed,
                        backbone_kind=args.backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    ce, bce = nn.CrossEntropyLoss(), nn.BCELoss()
    best = 0.0
    tag = (f"logiccbm_k{K}_{args.n_neurons}n"
           f"{'_learngates' if args.learn_gates else ''}"
           f"{'' if args.backbone == 'resnet18' else '_' + args.backbone}")
    path = os.path.join(args.save_dir, f"{tag}.pt")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"===== {tag}: K={K} neurons={args.n_neurons} attr312={attr312} "
          f"fixed_gates={not args.learn_gates} concept_lam={args.concept_lam} "
          f"({args.epochs}ep) =====", flush=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        for img, cattr, y in tl:
            img, cattr, y = img.to(device), cattr.to(device).float(), y.to(device)
            logits, c = model(img)
            loss = ce(logits, y) + args.concept_lam * bce(c.clamp(1e-6, 1 - 1e-6), cattr)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, vl, device)
        best = max(best, acc)
        if acc >= best:
            torch.save({"state_dict": model.state_dict(), "K": K,
                        "n_neurons": args.n_neurons, "fixed_gates": not args.learn_gates,
                        "attr312": attr312, "backbone": args.backbone, "acc": acc}, path)
        print(f"    epoch {ep:3d}/{args.epochs} | loss {loss.item():.3f} | "
              f"test {acc*100:.2f}% | best {best*100:.2f}%", flush=True)
    print(f"  DONE {tag}: best {best*100:.2f}%  saved {path}", flush=True)


def cmd_eval(args):
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CUBLogicCBM(K=ck["K"], n_neurons=ck["n_neurons"],
                        fixed_gates=ck["fixed_gates"],
                        backbone_kind=ck.get("backbone", "resnet18")).to(device)
    model.load_state_dict(ck["state_dict"])
    _isz = 299 if ck.get("backbone", "resnet18") == "inception_v3" else 224
    vl = DataLoader(_cub._CUBImages("test", False, attr312=ck.get("attr312", False), image_size=_isz),
                    batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"test accuracy: {evaluate(model, vl, device)*100:.2f}%  (saved {ck.get('acc',0)*100:.2f}%)")


# --------------------------------------------------------------- rule extraction
_TRIVIAL_GATES = {0, 15}   # constant F / T neurons carry no concept information


def load_model(ckpt, device="cpu"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = CUBLogicCBM(K=ck["K"], n_neurons=ck["n_neurons"],
                        fixed_gates=ck["fixed_gates"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def extract_rules(model, cls, names, topn=None, drop_trivial=True):
    """Per-class LogicCBM rule = weighted sum of the 250 logic predicates.

    Returns a list of dicts ``{weight, gate, gate_id, a, b, predicate, neuron}``
    sorted by descending |weight| (the classifier row for ``cls``). ``predicate``
    is the verbalized ``gate(a, b)`` using the concept NAMES. Positive weight =>
    the predicate supports the class; negative => it votes against."""
    w = model.classifier.weight[cls].detach().cpu()          # (n_neurons,)
    gate_ids = model.logic.hard_gate_ids().detach().cpu()     # (n_neurons,)
    a_idx = model.logic.a_idx.detach().cpu()
    b_idx = model.logic.b_idx.detach().cpu()
    rules = []
    for j in range(w.numel()):
        gid = int(gate_ids[j])
        if drop_trivial and gid in _TRIVIAL_GATES:
            continue
        a_nm, b_nm = names[int(a_idx[j])], names[int(b_idx[j])]
        rules.append({
            "neuron": j, "weight": float(w[j]), "gate_id": gid,
            "gate": GATE_STR[gid], "a": a_nm, "b": b_nm,
            "predicate": GATE_STR[gid].format(a=a_nm, b=b_nm),
        })
    rules.sort(key=lambda r: -abs(r["weight"]))
    return rules if topn is None else rules[:topn]


def cmd_rules(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.ckpt, device)
    names = (_cub.load_names_312() if ck.get("attr312", False)
             else _cub._load_attr_groups()[0])
    species = _cub._load_class_names() if hasattr(_cub, "_load_class_names") else None
    n_classes = model.classifier.weight.size(0)
    class_list = [args.cls] if args.cls is not None else range(min(args.max_classes, n_classes))
    for cls in class_list:
        title = f"class {cls}" + (f" ({species[cls]})" if species else "")
        print(f"\n===== {title} =====")
        for r in extract_rules(model, cls, names, topn=args.topn):
            sign = "+" if r["weight"] >= 0 else "-"
            print(f"  {sign}{abs(r['weight']):.3f}  [{r['gate_id']:2d}]  {r['predicate']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--K", type=int, default=112)
    p.add_argument("--n-neurons", type=int, default=250)
    p.add_argument("--attr312", action="store_true",
                   help="use the raw 312 class-level CUB attributes (forces K=312); default = 112 denoised concepts")
    p.add_argument("--learn-gates", action="store_true",
                   help="learn the per-neuron gate (softmax over 16); default = fixed random gates")
    p.add_argument("--concept-lam", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--backbone", choices=["resnet18", "inception_v3"], default="resnet18",
                   help="feature backbone. inception_v3 = Koh-CBM/LogicCBM standard (299px, 2048-d)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=os.path.join(_HERE, "saved"))
    p.set_defaults(func=train)
    p = sub.add_parser("eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("rules")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cls", type=int, default=None, help="one class id; default = first --max-classes")
    p.add_argument("--topn", type=int, default=12)
    p.add_argument("--max-classes", type=int, default=5)
    p.set_defaults(func=cmd_rules)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
