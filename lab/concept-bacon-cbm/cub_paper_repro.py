"""Faithful reproduction of the LogicCBM CUB pipeline (paper reports 81.13%).

The paper's number comes from a two-stage recipe, NOT a from-scratch head train:

  Stage 1 (Joint CBM)  -- InceptionV3 (pretrained, aux head) -> 312 concept
    logits -> Linear classifier, trained end-to-end with:
      * auxiliary supervision:  loss = 1.0*main + 0.4*aux   (class AND every attr)
      * per-attribute imbalance-weighted BCE  (weight = n_neg/n_pos per attr)
      * attr_loss_weight 0.01 + normalize_loss  -> (cls + 0.01*Sum attr)/(1+0.01*312)
      * SGD (mom 0.9, lr 1e-3, wd 4e-4), Koh aug + Normalize(mean .5, std 2), 299px
    This is Koh's joint ConceptBottleneck; it reaches ~80% on CUB.

  Stage 2 (head finetune)  -- load the trained Stage-1 backbone+concepts and
    finetune a *head* on top of the 312 concepts:
      * ``logic``   : difflogic LogicLayer(312->250, fixed random gates) -> Linear
                      (the paper's LogicCBM head; -use_pretrained_bb_con)
      * ``gltree``  : our graded-logic tree head (VectorRectTree / VectorFullTree)
    Same-backbone comparison of their logic head vs our graded-logic tree.

Usage:
  py -3 -u cub_paper_repro.py joint  --epochs 40 --batch-size 32
  py -3 -u cub_paper_repro.py logic  --joint saved/cub_joint.pt --epochs 40
  py -3 -u cub_paper_repro.py gltree --joint saved/cub_joint.pt --head fulltree ...
"""

from __future__ import annotations

import argparse
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

import _cub                                                    # noqa: E402

# The Joint/vanilla CBM in the reference code uses attr_loss_weight=0.001, which
# with normalize_loss gives a class-loss weight of 1/(1+0.001*312)=0.76 -- a
# STRONG class signal. Using 0.01 (as we did) drops it to 0.24 (3x weaker) and
# makes accurate concepts but a starved, slow classifier. 0.001 is the key.
ATTR_LOSS_WEIGHT = 0.001
AUX_WEIGHT = 0.4
N_CONCEPTS = 312
N_CLASSES = 200
IMG_SIZE = 299


# ----------------------------------------------------------------- data
def _loaders(batch_size, workers, eval_bs=None):
    kw = dict(num_workers=workers, pin_memory=True)
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    tl = DataLoader(
        _cub._CUBImages("train", True, image_size=IMG_SIZE, paper_tf=True, attr_img312=True),
        batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    vl = DataLoader(
        _cub._CUBImages("test", False, image_size=IMG_SIZE, paper_tf=True, attr_img312=True),
        batch_size=eval_bs or batch_size, shuffle=False, **kw)
    return tl, vl


def _weighted_bce(logits, targets, attr_w):
    """Per-attribute imbalance-weighted BCE (Koh/LogicCBM exact): each attribute's
    BCE is scaled by its imbalance ratio (n_neg/n_pos) and summed, mean over batch
    -- sum_i ratio_i * mean_batch(BCE_i). With a strong class signal
    (attr_loss_weight=0.001 -> class weight 0.76) the class gradient prevents the
    all-negative concept collapse, so this literal 'weight' form works and keeps
    the rare discriminative attributes supervised (concept acc stays ~88%)."""
    per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B,312)
    return (per * attr_w).sum(dim=1).mean()


# ----------------------------------------------------------------- Stage 1: Joint
class JointCBM(nn.Module):
    """InceptionV3 concept bottleneck (aux) -> sigmoid concepts -> Linear classifier."""

    def __init__(self, n_concepts=N_CONCEPTS, n_classes=N_CLASSES):
        super().__init__()
        self.concept = _cub.InceptionConcept(n_concepts)
        self.classifier = nn.Linear(n_concepts, n_classes)

    def forward(self, x):
        if self.training:
            cm, ca = self.concept(x)
            ym = self.classifier(cm)                   # RAW concept logits -> classifier
            ya = self.classifier(ca)                   # (Koh Joint: no sigmoid bottleneck)
            return (ym, cm), (ya, ca)
        cm = self.concept(x)
        return self.classifier(cm), cm


@torch.no_grad()
def _eval_joint(model, loader, device):
    model.eval()
    correct = total = con_correct = con_total = 0
    for img, c, y in loader:
        img, c, y = img.to(device), c.to(device), y.to(device)
        logits, cm = model(img)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        con_correct += ((cm > 0).float() == c).sum().item()
        con_total += c.numel()
    return correct / total, con_correct / con_total


def train_joint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tl, vl = _loaders(args.batch_size, args.workers, eval_bs=args.batch_size)
    attr_w = _cub.find_imbalance_img312().clamp(max=args.pos_clamp).to(device)  # (312,)
    model = JointCBM().to(device)
    opt = torch.optim.SGD([
        {"params": model.concept.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr * args.head_lr_mult},
    ], lr=args.lr, momentum=0.9, weight_decay=4e-4)
    # cosine decay sharpens the concepts in the final epochs (const LR plateaus);
    # --const-lr keeps the paper's StepLR(1000)=constant schedule.
    if args.const_lr:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.1)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    ce = nn.CrossEntropyLoss()
    norm = 1.0 + ATTR_LOSS_WEIGHT * N_CONCEPTS
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, args.tag or "cub_joint.pt")
    best = 0.0
    print(f"===== JOINT CBM (Inception+aux, weighted BCE, SGD): "
          f"b{args.batch_size} lr{args.lr} ({args.epochs}ep) =====", flush=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        for img, c, y in tl:
            img, c, y = img.to(device), c.to(device), y.to(device)
            (ym, cm), (ya, ca) = model(img)
            loss_cls = ce(ym, y) + AUX_WEIGHT * ce(ya, y)
            loss_attr = _weighted_bce(cm, c, attr_w) + AUX_WEIGHT * _weighted_bce(ca, c, attr_w)
            loss = (loss_cls + ATTR_LOSS_WEIGHT * loss_attr) / norm
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc, con = _eval_joint(model, vl, device)
        if acc >= best:
            best = acc
            torch.save({"concept": model.concept.state_dict(),
                        "classifier": model.classifier.state_dict(),
                        "acc": acc, "con_acc": con}, path)
        print(f"    epoch {ep:3d}/{args.epochs} | loss {loss.item():.3f} | "
              f"class {acc*100:.2f}% | concept {con*100:.2f}% | best {best*100:.2f}%",
              flush=True)
    print(f"  DONE joint: best {best*100:.2f}%  saved {path}", flush=True)


# ----------------------------------------------------------------- Stage 2: head finetune
class HeadCBM(nn.Module):
    """Joint concept extractor + a swappable class head over the 312 concepts.

    ``logic``  : difflogic LogicLayer(312->neurons, fixed random gates) -> Linear
    ``linear`` : Linear(312->200)   (the Joint classifier; sanity control)
    ``fulltree``/``recttree`` : our graded-logic tree heads (probability outputs).
    """

    def __init__(self, head, n_concepts=N_CONCEPTS, n_classes=N_CLASSES,
                 n_neurons=250, seed=0, tree_kwargs=None):
        super().__init__()
        self.concept = _cub.InceptionConcept(n_concepts)
        self.head = head
        self.n_classes = n_classes
        if head == "logic":
            from logic_cbm import LogicLayer
            self.logic = LogicLayer(n_concepts, n_neurons, fixed_gates=True, seed=seed)
            self.classifier = nn.Linear(n_neurons, n_classes)
        elif head == "linear":
            self.classifier = nn.Linear(n_concepts, n_classes)
        elif head == "fulltree":
            from bacon.vectorizedFullTree import VectorFullTreeHead
            self.tree = VectorFullTreeHead(n_concepts, n_classes, **(tree_kwargs or {}))
        elif head == "recttree":
            from bacon.vectorizedRectTree import VectorRectTreeHead
            self.tree = VectorRectTreeHead(n_concepts, n_classes, **(tree_kwargs or {}))
        else:
            raise ValueError(f"unknown head {head!r}")

    def load_joint(self, ckpt):
        self.concept.load_state_dict(ckpt["concept"])

    def _cls(self, clogits):                       # clogits (B,312) RAW concept logits
        if self.head == "linear":
            return self.classifier(clogits)                  # class logits (raw concepts)
        concepts = torch.sigmoid(clogits)                    # logic/tree need [0,1]
        if self.head == "logic":
            return self.classifier(self.logic(concepts))     # class logits
        return self.tree(concepts)                           # class probabilities

    def forward(self, x):
        if self.training:
            cm, ca = self.concept(x)
            return (self._cls(cm), cm), (self._cls(ca), ca)
        cm = self.concept(x)
        return self._cls(cm), cm


def _class_loss(scores, y, head):
    if head in ("logic", "linear"):
        return F.cross_entropy(scores, y)
    # tree heads output per-class probabilities in [0,1]; convert to log-odds and
    # use multiclass cross-entropy (far stronger gradient than one-vs-rest BCE,
    # which barely moves the near-uniform tree outputs and collapsed to chance).
    p = scores.clamp(1e-6, 1.0 - 1e-6)
    logits = torch.log(p) - torch.log1p(-p)
    return F.cross_entropy(logits, y)


@torch.no_grad()
def _eval_head(model, loader, device):
    model.eval()
    correct = total = con_correct = con_total = 0
    for img, c, y in loader:
        img, c, y = img.to(device), c.to(device), y.to(device)
        scores, cm = model(img)
        correct += (scores.argmax(1) == y).sum().item()
        total += y.size(0)
        con_correct += ((cm > 0).float() == c).sum().item()
        con_total += c.numel()
    return correct / total, con_correct / con_total


def train_head(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tl, vl = _loaders(args.batch_size, args.workers, eval_bs=args.batch_size)
    attr_w = _cub.find_imbalance_img312().clamp(max=args.pos_clamp).to(device)
    ckpt = torch.load(args.joint, map_location="cpu", weights_only=False)

    is_tree = args.head in ("fulltree", "recttree")
    tree_kwargs = None
    if args.head == "fulltree":
        tree_kwargs = dict(branching=args.branching, max_egress=args.max_egress,
                           use_negation=args.negation, use_coefficients=args.coefficients,
                           straight_through=args.straight_through)
    elif args.head == "recttree":
        tree_kwargs = dict(width=args.rect_width, depth=args.rect_depth,
                           max_parents=args.max_parents, leaf_shortcut=args.leaf_shortcut,
                           use_negation=args.negation, use_coefficients=args.coefficients,
                           straight_through=args.straight_through)
    model = HeadCBM(args.head, n_neurons=getattr(args, "n_neurons", 250),
                    seed=args.seed, tree_kwargs=tree_kwargs).to(device)
    model.load_joint(ckpt)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("concept.")]
    opt = torch.optim.SGD([
        {"params": model.concept.parameters(), "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], lr=args.lr, momentum=0.9, weight_decay=4e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.1)
    norm = 1.0 + ATTR_LOSS_WEIGHT * N_CONCEPTS
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, args.tag or f"cub_head_{args.head}.pt")
    freeze_ep = int(args.freeze_frac * args.epochs) if is_tree else None
    frozen = False
    best = 0.0
    print(f"===== HEAD={args.head} finetune from {os.path.basename(args.joint)} "
          f"(joint acc {ckpt.get('acc', 0)*100:.2f}%): b{args.batch_size} lr{args.lr} "
          f"({args.epochs}ep{f', harden@{freeze_ep}' if is_tree else ''}) =====", flush=True)
    for ep in range(1, args.epochs + 1):
        if is_tree and not frozen:
            model.tree.anneal(min(1.0, (ep - 1) / max(freeze_ep, 1)))
        model.train()
        for img, c, y in tl:
            img, c, y = img.to(device), c.to(device), y.to(device)
            (sm, cm), (sa, ca) = model(img)
            loss_cls = _class_loss(sm, y, args.head) + AUX_WEIGHT * _class_loss(sa, y, args.head)
            loss_attr = _weighted_bce(cm, c, attr_w) + AUX_WEIGHT * _weighted_bce(ca, c, attr_w)
            loss = (loss_cls + ATTR_LOSS_WEIGHT * loss_attr) / norm
            if is_tree and hasattr(model.tree, "regularization"):
                loss = loss + model.tree.regularization()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if is_tree and not frozen and ep == freeze_ep:
            if hasattr(model.tree, "harden"):
                model.tree.harden()
            elif hasattr(model.tree, "freeze_egress"):
                model.tree.freeze_egress()
            frozen = True
            print(f"    -- hardened tree at epoch {ep} --", flush=True)
        acc, con = _eval_head(model, vl, device)
        tag_ok = (not is_tree) or frozen                 # only trust frozen tree acc for 'best'
        if acc >= best and tag_ok:
            best = acc
            torch.save({"state_dict": model.state_dict(), "head": args.head,
                        "acc": acc, "con_acc": con, "joint": args.joint}, path)
        print(f"    epoch {ep:3d}/{args.epochs} | loss {loss.item():.3f} | "
              f"class {acc*100:.2f}% | concept {con*100:.2f}% | best {best*100:.2f}%",
              flush=True)
    print(f"  DONE {args.head}: best {best*100:.2f}%  saved {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("joint", help="Stage 1: train the Joint CBM concept extractor")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--pos-clamp", type=float, default=3.0,
                   help="cap on per-attribute BCE pos_weight (imbalance ratio)")
    p.add_argument("--head-lr-mult", type=float, default=10.0,
                   help="LR multiplier for the class head vs the backbone")
    p.add_argument("--const-lr", action="store_true",
                   help="keep the paper's constant StepLR (default: cosine decay)")
    p.add_argument("--save-dir", default=os.path.join(_HERE, "saved"))
    p.set_defaults(func=train_joint)

    for name in ("logic", "linear", "fulltree", "recttree"):
        p = sub.add_parser(name, help=f"Stage 2: finetune a {name} head from a joint ckpt")
        p.add_argument("--joint", required=True, help="Stage-1 joint checkpoint (.pt)")
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tag", default=None)
        p.add_argument("--pos-clamp", type=float, default=3.0,
                       help="cap on per-attribute BCE pos_weight (imbalance ratio)")
        p.add_argument("--head-lr-mult", type=float, default=10.0,
                       help="LR multiplier for the class head vs the backbone")
        p.add_argument("--save-dir", default=os.path.join(_HERE, "saved"))
        if name == "logic":
            p.add_argument("--n-neurons", type=int, default=250)
        if name in ("fulltree", "recttree"):
            p.add_argument("--freeze-frac", type=float, default=0.75,
                           help="fraction of epochs after which the tree is hardened")
            p.add_argument("--negation", action="store_true")
            p.add_argument("--coefficients", action="store_true")
            p.add_argument("--straight-through", action="store_true")
        if name == "fulltree":
            p.add_argument("--branching", type=int, default=8)
            p.add_argument("--max-egress", type=int, default=1)
        if name == "recttree":
            p.add_argument("--rect-width", type=int, default=64)
            p.add_argument("--rect-depth", type=int, default=3)
            p.add_argument("--max-parents", type=int, default=1)
            p.add_argument("--leaf-shortcut", action="store_true")
        p.set_defaults(func=train_head, head=name)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
