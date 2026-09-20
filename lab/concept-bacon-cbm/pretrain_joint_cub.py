"""Stage-1 joint CBM pretrain (LogicCBM -use_pretrained_bb_con analogue).

Trains a plain Joint CBM -- backbone -> Linear(feat, K) concepts -> sigmoid ->
Linear(K, n_classes) classifier -- end-to-end with class CE + per-attribute
imbalance-weighted concept BCE. Saves the WARMED backbone + concept-layer weights
so the graded-logic tree head (run_cub_fulltree.py --init-from) starts from good
concepts instead of ImageNet init (kills the slow early warmup, higher ceiling).
"""
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


class JointCBM(nn.Module):
    """backbone -> concept logits -> sigmoid -> linear class head (joint CBM)."""

    def __init__(self, K=312, n_classes=200, backbone_kind="inception_v3", use_aux=False):
        super().__init__()
        self.use_aux = bool(use_aux) and backbone_kind == "inception_v3"
        if self.use_aux:
            # Koh-CBM InceptionV3 with the AUX classifier retargeted to predict
            # concepts: deep supervision on both the 2048-d main and 768-d aux
            # branches (train -> (main, aux) concept logits; eval -> main only).
            self.encoder = _cub.InceptionConcept(K)
            self.backbone_size = 299
        else:
            self.backbone, feat_dim, self.backbone_size = _cub.make_backbone(backbone_kind)
            self.concept = nn.Linear(feat_dim, K)
        self.classifier = nn.Linear(K, n_classes)

    def forward(self, x):
        if self.use_aux:
            if self.training:
                main_logit, aux_logit = self.encoder(x)
                c, ac = torch.sigmoid(main_logit), torch.sigmoid(aux_logit)
                return self.classifier(c), c, self.classifier(ac), ac
            c = torch.sigmoid(self.encoder(x))
            return self.classifier(c), c
        c = torch.sigmoid(self.concept(self.backbone(x)))
        return self.classifier(c), c

    # --- phase-2-compatible state dicts (feature-extractor backbone + concept fc) ---
    def export_backbone_concept(self):
        """Return (backbone_sd, concept_sd) matching a plain make_backbone feature
        extractor + Linear(feat, K), so run_cub_fulltree --init-from can strict-load
        them regardless of whether the aux head was used during pretraining."""
        if self.use_aux:
            net_sd = self.encoder.net.state_dict()
            backbone_sd = {k: v for k, v in net_sd.items()
                           if not k.startswith("fc.") and not k.startswith("AuxLogits.")}
            concept_sd = self.encoder.net.fc.state_dict()
        else:
            backbone_sd = self.backbone.state_dict()
            concept_sd = self.concept.state_dict()
        return backbone_sd, concept_sd


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    cbin = cn = 0
    for img, c, y in loader:
        img, y = img.to(device), y.to(device)
        logits, cpred = model(img)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
        cb = ((cpred > 0.5).float() == c.to(device)).float()
        cbin += cb.sum().item()
        cn += cb.numel()
    return correct / max(total, 1), cbin / max(cn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["resnet18", "inception_v3"], default="inception_v3")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concept-lam", type=float, default=0.03,
                    help="weight on the (imbalance-weighted) concept BCE")
    ap.add_argument("--concept-imbalance", action="store_true",
                    help="per-attribute n_neg/n_pos weighting (LogicCBM weighted_loss multiple)")
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help=">0 -> AdamW with this decay (regularize the overfitting backbone)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CE label smoothing (regularization)")
    ap.add_argument("--strong-aug", action="store_true",
                    help="RandAugment + RandomErasing on the train images")
    ap.add_argument("--use-aux", action="store_true",
                    help="InceptionV3 auxiliary-head deep supervision on class+concepts "
                         "(0.4x weight; the Koh-CBM / LogicCBM trick, inception_v3 only)")
    ap.add_argument("--save-dir", default=os.path.join(_HERE, "saved"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 312
    image_size = 299 if args.backbone == "inception_v3" else 224
    eval_bs = 64 if args.backbone == "inception_v3" else 128

    model = JointCBM(K, 200, backbone_kind=args.backbone, use_aux=args.use_aux).to(device)

    pos_weight = None
    iwtag = ""
    if args.concept_imbalance:
        pos_weight = _cub.find_imbalance_img312().to(device)
        iwtag = "_iw"

    os.makedirs(args.save_dir, exist_ok=True)
    regtag = ""
    if args.weight_decay > 0 or args.label_smoothing > 0 or args.strong_aug:
        regtag = f"_reg"
    auxtag = "_aux" if model.use_aux else ""
    path = os.path.join(
        args.save_dir,
        f"cub_joint_pretrain_{args.backbone}_k{K}{iwtag}{regtag}{auxtag}_{args.epochs}ep.pt")
    if os.path.exists(path):
        print(f"checkpoint exists, SKIPPING ({path})", flush=True)
        return

    tl = DataLoader(_cub._CUBImages("train", True, attr312=True, image_size=image_size,
                                    strong_aug=args.strong_aug),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False, attr312=True, image_size=image_size),
                    batch_size=eval_bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    # backbone (low lr) vs heads (concept fc + classifier, high lr). With the aux
    # head the concept projections live inside the encoder, so split by name.
    if model.use_aux:
        head_ids = set()
        for m in (model.classifier, model.encoder.net.fc, model.encoder.net.AuxLogits.fc):
            head_ids |= {id(p) for p in m.parameters()}
        bb = [p for p in model.parameters() if id(p) not in head_ids]
        heads = [p for p in model.parameters() if id(p) in head_ids]
    else:
        bb_ids = {id(p) for p in model.backbone.parameters()}
        bb = [p for p in model.parameters() if id(p) in bb_ids]
        heads = [p for p in model.parameters() if id(p) not in bb_ids]
    if args.weight_decay > 0:
        opt = torch.optim.AdamW([{"params": bb, "lr": args.backbone_lr},
                                 {"params": heads, "lr": args.head_lr}],
                                weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam([{"params": bb, "lr": args.backbone_lr},
                                {"params": heads, "lr": args.head_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    def concept_bce(cpred, c):
        if pos_weight is not None:
            p = cpred.clamp(1e-6, 1.0 - 1e-6)
            return -(pos_weight * c * p.log() + (1.0 - c) * (1.0 - p).log()).mean()
        return F.binary_cross_entropy(cpred, c)

    print(f"\n===== JOINT PRETRAIN {args.backbone} K{K}{iwtag} lam{args.concept_lam} "
          f"({args.epochs}ep, b{args.batch_size}){' +aux' if model.use_aux else ''} =====", flush=True)
    best = 0.0
    best_state = None
    for ep in range(args.epochs):
        model.train()
        run = tot = 0
        for img, c, y in tl:
            img, y, c = img.to(device), y.to(device), c.to(device)
            out = model(img)
            if len(out) == 4:
                logits, cpred, aux_logits, aux_c = out
            else:
                logits, cpred = out
                aux_logits = aux_c = None
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            if aux_logits is not None:
                loss = loss + 0.4 * F.cross_entropy(aux_logits, y,
                                                    label_smoothing=args.label_smoothing)
            if args.concept_lam > 0.0:
                loss = loss + args.concept_lam * concept_bce(cpred, c)
                if aux_c is not None:
                    loss = loss + args.concept_lam * 0.4 * concept_bce(aux_c, c)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * y.numel(); tot += y.numel()
        sched.step()
        acc, cacc = evaluate(model, vl, device)
        if acc > best:
            best = acc
            backbone_sd, concept_sd = model.export_backbone_concept()
            best_state = {"backbone": backbone_sd,
                          "concept": concept_sd,
                          "classifier": model.classifier.state_dict()}
        print(f"    epoch {ep + 1:3d}/{args.epochs} | loss {run / tot:.3f} | "
              f"class {acc * 100:.2f}% | concept {cacc * 100:.2f}% | best {best * 100:.2f}%",
              flush=True)

    torch.save({**best_state, "K": K, "backbone_kind": args.backbone,
                "class_acc": best}, path)
    print(f"  DONE joint pretrain: class {best * 100:.2f}%  saved {path}", flush=True)


if __name__ == "__main__":
    main()
