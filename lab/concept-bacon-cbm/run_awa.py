r"""AwA2 BACON graded-logic CBM runner (LogicCBM-comparable).

Two stages, mirroring the CUB pipeline:

  joint : InceptionV3 -> 85 concept logits -> 50-class linear, trained end-to-end
          with CE(class) + attr_lam * imbalance-weighted BCE(concept vs the
          class-level 85 predicates). Saves a {backbone, concept, class_acc}
          checkpoint = the same format ``cub_emergent.train_one --init-from`` reads.

  tree  : warm-start from the joint checkpoint and train our recttree graded-logic
          head over the 85 concepts (50 trees), reusing the validated
          ``cub_emergent.train_one`` machinery (freeze / harden / L0 / confusion-ovr).

    py -3 run_awa.py joint --epochs 60 --batch-size 32
    py -3 run_awa.py tree  --init-from saved/awa_joint_inception_v3_k85.pt \
        --rect-width 64 --rect-depth 3 --max-parents 2 --edge-l0 1e-4 --edge-l0-warmup 40 \
        --coefficients --negation --confusion-ovr --epochs 200
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
import awa_data                                                # noqa: E402

_K = 85
_N_CLASSES = 50


# --------------------------------------------------------------- joint CBM
class AwAJoint(nn.Module):
    """InceptionV3 backbone -> 85 concept logits -> 50-class linear (joint CBM).

    ``backbone`` + ``concept`` submodule names/keys match ``CUBEmergent`` so the
    saved state dicts load directly via ``train_one --init-from``."""

    def __init__(self, backbone_kind="inception_v3"):
        super().__init__()
        self.backbone, feat_dim, self.input_size = _cub.make_backbone(backbone_kind)
        self.concept = nn.Linear(feat_dim, _K)
        self.classifier = nn.Linear(_K, _N_CLASSES)

    def forward(self, x):
        f = self.backbone(x)
        c_logit = self.concept(f)
        y = self.classifier(torch.sigmoid(c_logit))
        return y, c_logit


@torch.no_grad()
def _eval_joint(model, loader, device):
    model.eval()
    correct = total = 0
    c_correct = c_total = 0
    for img, c, y in loader:
        img, y, c = img.to(device), y.to(device), c.to(device)
        yl, cl = model(img)
        correct += (yl.argmax(1) == y).sum().item()
        total += y.numel()
        c_correct += ((cl > 0).float() == c).sum().item()
        c_total += c.numel()
    return correct / max(total, 1), c_correct / max(c_total, 1)


def run_joint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 299 if args.backbone == "inception_v3" else 224
    tl = DataLoader(awa_data._AwAImages("train", True, image_size=image_size),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(awa_data._AwAImages("test", False, image_size=image_size),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True)
    model = AwAJoint(args.backbone).to(device)
    pos_weight = awa_data.find_imbalance().to(device)          # (85,) n_neg/n_pos
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for img, c, y in tl:
            img, y, c = img.to(device), y.to(device), c.to(device)
            yl, cl = model(img)
            cls_loss = F.cross_entropy(yl, y)
            # LogicCBM-faithful: SUM the per-attribute BCE (not mean) then
            # normalize by (1 + attr_lam*K) so concepts get a real gradient
            # (mean-BCE * 0.01 was negligible -> concepts never learned, ~51%).
            con_loss = F.binary_cross_entropy_with_logits(
                cl, c, pos_weight=pos_weight, reduction="none").sum(dim=1).mean()
            loss = (cls_loss + args.attr_lam * con_loss) / (1.0 + args.attr_lam * _K)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        acc, cacc = _eval_joint(model, vl, device)
        print(f"  epoch {ep + 1}/{args.epochs} | loss {tot / len(tl):.3f} | "
              f"class {acc * 100:.2f}% | concept {cacc * 100:.2f}%", flush=True)
        if acc >= best:
            best = acc
            torch.save({"backbone": model.backbone.state_dict(),
                        "concept": model.concept.state_dict(),
                        "classifier": model.classifier.state_dict(),
                        "class_acc": acc, "K": _K, "n_classes": _N_CLASSES,
                        "backbone_kind": args.backbone}, args.out)
    print(f"  DONE joint: best class {best * 100:.2f}%  saved {args.out}", flush=True)


# --------------------------------------------------------------- tree stage
def run_tree(args):
    from cub_emergent import train_one                         # noqa: E402
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 299 if args.backbone == "inception_v3" else 224
    tl = DataLoader(awa_data._AwAImages("train", True, image_size=image_size),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(awa_data._AwAImages("test", False, image_size=image_size),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True)
    pos_weight = awa_data.find_imbalance()
    rect_layer_widths = None
    # NOTE: train_one must accept n_species (generalized from the CUB-hardcoded 200).
    acc, model = train_one(
        _K, tl, vl, device, args.epochs, seed=args.seed, head="recttree",
        n_species=_N_CLASSES,
        loss_mode="hybrid", ovr_weight=0.3,
        concept_lam=args.concept_lam, concept_pos_weight=pos_weight,
        confusion_ovr=args.confusion_ovr, confusion_hardness=args.confusion_hardness,
        rect_width=args.rect_width, rect_depth=args.rect_depth,
        rect_max_parents=args.max_parents, rect_edge_l0_lam=args.edge_l0,
        rect_edge_l0_warmup=args.edge_l0_warmup,
        rect_binarize_lam=args.binarize_lam, rect_binarize_warmup=args.binarize_warmup,
        fulltree_coefficients=args.coefficients, fulltree_negation=args.negation,
        early_stop_patience=args.early_stop_patience, min_freeze_frac=args.min_freeze_frac,
        backbone_kind=args.backbone, init_from=args.init_from, freeze_encoder=args.freeze_encoder,
        harden=True)
    st = "_bin" if args.binarize_warmup > 0 else ""
    ut = f"_{args.tag}" if args.tag else ""
    tag = f"awa_recttree_w{args.rect_width}_l0{args.edge_l0:g}{st}_s{args.seed}{ut}"
    path = os.path.join(_HERE, "saved", f"{tag}_{args.backbone}_{args.epochs}ep.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "K": _K, "n_classes": _N_CLASSES,
                "head": "recttree", "backbone": args.backbone, "attr312": False,
                "rect_width": args.rect_width, "rect_depth": args.rect_depth,
                "rect_layer_widths": rect_layer_widths,
                "rect_max_parents": args.max_parents,
                "rect_leaf_shortcut": False}, path)
    print(f"  DONE tree: acc {acc * 100:.2f}%  saved {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)

    j = sub.add_parser("joint")
    j.add_argument("--backbone", default="inception_v3", choices=["resnet18", "inception_v3"])
    j.add_argument("--epochs", type=int, default=60)
    j.add_argument("--batch-size", type=int, default=32)
    j.add_argument("--lr", type=float, default=1e-3)
    j.add_argument("--wd", type=float, default=4e-4)
    j.add_argument("--attr-lam", type=float, default=0.01)
    j.add_argument("--workers", type=int, default=4)
    j.add_argument("--out", default=os.path.join(_HERE, "saved", "awa_joint_inception_v3_k85.pt"))

    t = sub.add_parser("tree")
    t.add_argument("--backbone", default="inception_v3", choices=["resnet18", "inception_v3"])
    t.add_argument("--init-from", required=True)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--concept-lam", type=float, default=0.03)
    t.add_argument("--rect-width", type=int, default=64)
    t.add_argument("--rect-depth", type=int, default=3)
    t.add_argument("--max-parents", type=int, default=2)
    t.add_argument("--edge-l0", type=float, default=1e-4)
    t.add_argument("--edge-l0-warmup", type=int, default=40)
    t.add_argument("--binarize-lam", type=float, default=0.0)
    t.add_argument("--binarize-warmup", type=int, default=0)
    t.add_argument("--coefficients", action="store_true")
    t.add_argument("--negation", action="store_true")
    t.add_argument("--confusion-ovr", action="store_true")
    t.add_argument("--confusion-hardness", type=float, default=1.0)
    t.add_argument("--early-stop-patience", type=int, default=40)
    t.add_argument("--min-freeze-frac", type=float, default=0.4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--tag", default="")
    # ANNOTATED mode (default): freeze the supervised concept encoder so the tree
    # reasons over faithful annotated concepts (prevents concept leakage).
    t.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true", default=True)
    t.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")

    args = ap.parse_args()
    if args.stage == "joint":
        run_joint(args)
    else:
        run_tree(args)


if __name__ == "__main__":
    main()
