"""
Shared helpers for the CelebA table cells (END-TO-END fine-tuned ResNet-18).

Protocol from the CREAM paper (arXiv:2506.05014, App. C.1 "Smile Detection"):
  * Task  : predict "Smiling"  -> binary classification (L = 2).
  * K = 7 concept attributes:
        Arched_Eyebrows, Bags_Under_Eyes, Double_Chin, Mouth_Slightly_Open,
        Narrow_Eyes, High_Cheekbones, Rosy_Cheeks
  * A_Y   : full (all 7 concepts directly connected to the task).
  * Backbone: ImageNet ResNet-18, fine-tuned on a 5K-image subset.
Reuses the generalized model classes (backbone-injectable, class count from
spec.A_Y) exactly like the CUB harness.

DATA LAYOUT (point CELEBA_ROOT at a dir containing):
    img_align_celeba/000001.jpg ...
    list_attr_celeba.txt          (header count, 40 attr names, then rows)
    list_eval_partition.txt       (filename  0/1/2  = train/val/test)
Obtain from the Kaggle mirror `jessicali9530/celeba-dataset` (has the aligned
images + attribute/partition files) or the torchvision CelebA layout.
"""

from __future__ import annotations

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

import _bench                                                   # sets sys.path
from models import BlackBox, CtrueY, SoftCBM, SoftCBMSC, CREAM, BaconCBM  # noqa: E402

# Root of the CelebA files (override via CELEBA_ROOT env var).
CELEBA_ROOT = os.environ.get("CELEBA_ROOT", r"C:\School\datasets\celeba")

CONCEPTS = ["Arched_Eyebrows", "Bags_Under_Eyes", "Double_Chin",
            "Mouth_Slightly_Open", "Narrow_Eyes", "High_Cheekbones",
            "Rosy_Cheeks"]
TASK_ATTR = "Smiling"
N_TRAIN_SUBSET = 5000

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

LABELS = {
    "blackbox": "Black-box", "ctruey": "C_true->Y", "cbm": "CBM",
    "cbm+sc": "CBM+SC", "cream-wo-sc": "CREAM w/o SC", "cream": "CREAM",
    "ocbm": "OCBM", "ocbm-ft": "OCBM (fine-tuned)",
}

_TRAIN_TF = transforms.Compose([
    transforms.Resize(178), transforms.CenterCrop(178),
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
])
_TEST_TF = transforms.Compose([
    transforms.Resize(178), transforms.CenterCrop(178), transforms.Resize(128),
    transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
])


def _read_attr_file():
    """Parse list_attr_celeba.txt -> (filenames, attr_names, values(N,40) in {0,1})."""
    path = os.path.join(CELEBA_ROOT, "list_attr_celeba.txt")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    # line 0 = count, line 1 = attribute names
    attr_names = lines[1].split()
    files, vals = [], []
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        files.append(parts[0])
        vals.append([1.0 if v == "1" else 0.0 for v in parts[1:]])
    return files, attr_names, torch.tensor(vals, dtype=torch.float32)


def _read_partition():
    """filename -> partition (0 train / 1 val / 2 test)."""
    path = os.path.join(CELEBA_ROOT, "list_eval_partition.txt")
    part = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                part[p[0]] = int(p[1])
    return part


def _load_index():
    """Return dict with per-split (filenames, concepts(N,7), task(N,))."""
    files, names, vals = _read_attr_file()
    idx = {n: i for i, n in enumerate(names)}
    cidx = [idx[c] for c in CONCEPTS]
    tidx = idx[TASK_ATTR]
    part = _read_partition()
    out = {0: [[], [], []], 2: [[], [], []]}
    for i, fn in enumerate(files):
        pt = part.get(fn, 0)
        if pt == 1:                          # skip val
            continue
        bucket = out.setdefault(pt, [[], [], []])
        bucket[0].append(fn)
        bucket[1].append(vals[i, cidx])
        bucket[2].append(int(vals[i, tidx].item()))
    res = {}
    for pt, (fns, cs, ys) in out.items():
        res[pt] = (fns, torch.stack(cs) if cs else torch.zeros(0, len(CONCEPTS)),
                   torch.tensor(ys, dtype=torch.long))
    return res


class _CelebImages(Dataset):
    def __init__(self, files, concepts, tasks, train_aug):
        self.files = files
        self.concepts = concepts
        self.tasks = tasks
        self.tf = _TRAIN_TF if train_aug else _TEST_TF

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(CELEBA_ROOT, "img_align_celeba",
                                      self.files[i])).convert("RGB")
        return self.tf(img), self.concepts[i], int(self.tasks[i])


def build_spec():
    """CelebA ConceptSpec-like object (7 binary concepts, 2 task classes, full A_Y)."""
    K, L = len(CONCEPTS), 2
    spec = types.SimpleNamespace()
    spec.concept_names = list(CONCEPTS)
    spec.K = K
    spec.mutex_groups = []
    spec.binary_concepts = list(range(K))
    spec.index = {n: i for i, n in enumerate(CONCEPTS)}
    spec.A_Y = torch.ones(L, K)                     # full: every class reads all concepts
    # C-C block: one clique over all 7 face concepts (correlated), approximating
    # the paper's hierarchical graph (exact edges not fully specified).
    spec.class_on = {0: list(CONCEPTS)}
    # OCBM/BACON: interpretable smile tree over the 7 concepts (a smile raises the
    # cheekbones and typically parts the lips). class 1 = smiling, class 0 = not.
    _smile = "High_Cheekbones AND Mouth_Slightly_Open"
    spec.formulas = {1: _smile, 0: f"NOT ({_smile})"}
    spec.Y = None
    return spec


def _make_resnet():
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    return net


def _build_model(key, spec):
    if key == "blackbox":
        return BlackBox(feat_dim=512, n_classes=2, backbone=_make_resnet()), False, False
    if key == "cbm":
        return SoftCBM(spec, feat_dim=512, backbone=_make_resnet()), True, False
    if key == "cbm+sc":
        return SoftCBMSC(spec, feat_dim=512, dropout_p=0.8, backbone=_make_resnet()), True, True
    if key in ("cream", "cream-wo-sc"):
        return CREAM(spec, feat_dim=512, dropout_p=0.8, backbone=_make_resnet()), True, True
    if key == "ocbm":
        return BaconCBM(spec, feat_dim=512, backbone=_make_resnet()), True, False
    if key == "ocbm-ft":
        return BaconCBM(spec, feat_dim=512, finetune_logic=True,
                        backbone=_make_resnet()), True, False
    raise ValueError(key)


def _train(model, loader, device, epochs, lam, has_c,
           backbone_lr=1e-4, head_lr=1e-3, freeze_backbone=False):
    model.to(device).train()
    bb_ids = {id(p) for p in model.backbone.parameters()}
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        heads = [p for p in model.parameters() if id(p) not in bb_ids]
        opt = torch.optim.Adam(heads, lr=head_lr)
    else:
        bb = [p for p in model.parameters() if id(p) in bb_ids]
        heads = [p for p in model.parameters() if id(p) not in bb_ids]
        opt = torch.optim.Adam([{"params": bb, "lr": backbone_lr},
                                {"params": heads, "lr": head_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for _ in range(epochs):
        if freeze_backbone:
            model.backbone.eval()                    # freeze BN stats too
        for img, c, y in loader:
            img, c, y = img.to(device), c.to(device), y.to(device)
            logits, cprobs = model(img)
            loss = F.cross_entropy(logits, y)
            if has_c and cprobs is not None:
                loss = loss + lam * F.binary_cross_entropy(cprobs, c)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def _evaluate(model, loader, device, spec, has_side):
    model.eval()
    full_c = full_n = cpath_c = 0
    cacc_sum = cacc_n = 0
    for img, c, y in loader:
        img, c, y = img.to(device), c.to(device), y.to(device)
        logits, cprobs = model(img)
        full_c += (logits.argmax(1) == y).sum().item()
        full_n += y.numel()
        logits_cp = model(img, use_side=False)[0] if has_side else logits
        cpath_c += (logits_cp.argmax(1) == y).sum().item()
        if cprobs is not None:
            cacc_sum += ((cprobs > 0.5).float() == c).float().mean().item() * y.numel()
            cacc_n += y.numel()
    concept = (cacc_sum / cacc_n) if cacc_n else float("nan")
    return full_c / full_n, cpath_c / full_n, concept


def _run_ctruey(spec, device, splits, iters, epochs=40, seed=0):
    _, Ctr, ytr = splits[0]
    _, Cte, yte = splits[2]
    tl = DataLoader(TensorDataset(Ctr, ytr), batch_size=256, shuffle=True)
    accs = []
    for it in range(iters):
        _bench.set_seed(seed + it)
        model = CtrueY(spec).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(epochs):
            for c, y in tl:
                c, y = c.to(device), y.to(device)
                loss = F.cross_entropy(model(c)[0], y)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Cte.to(device))[0].argmax(1).cpu()
        accs.append((pred == yte).float().mean().item())
        print(f"  [iter {it + 1}/{iters}]  ACC_Y = {accs[-1] * 100:.2f}")
    return accs


def run_cell(model_key: str, iters: int, epochs: int = 40,
             batch_size: int = 256, lam: float = 1.0, seed: int = 0,
             num_workers: int = 6, train_subset: int = N_TRAIN_SUBSET,
             freeze_backbone: bool = False):
    device = _bench.get_device()
    spec = build_spec()
    splits = _load_index()
    label = LABELS.get(model_key, model_key)
    print(f"{label} / CelebA   iters={iters}  epochs={epochs}  K={spec.K}  "
          f"lam={lam}  train_subset={train_subset}  freeze={freeze_backbone}  "
          f"device={device}")

    if model_key == "ctruey":
        return _run_ctruey(spec, device, splits, iters, seed=seed), None

    tr_files, tr_c, tr_y = splits[0]
    te_files, te_c, te_y = splits[2]

    acc_y, acc_c = [], []
    for it in range(iters):
        _bench.set_seed(seed + it)
        # fresh 5K training subset per seed
        perm = torch.randperm(len(tr_files))[:train_subset]
        sub_files = [tr_files[i] for i in perm.tolist()]
        tl = DataLoader(_CelebImages(sub_files, tr_c[perm], tr_y[perm], True),
                        batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
        vl = DataLoader(_CelebImages(te_files, te_c, te_y, False),
                        batch_size=256, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
        model, has_c, has_side = _build_model(model_key, spec)
        _train(model, tl, device, epochs, lam, has_c, freeze_backbone=freeze_backbone)
        full, cpath, cacc = _evaluate(model, vl, device, spec, has_side)
        ay = cpath if model_key == "cream-wo-sc" else full
        acc_y.append(ay)
        if has_c:
            acc_c.append(cacc)
        cstr = f"  ACC_C = {cacc * 100:.2f}" if has_c else ""
        print(f"  [iter {it + 1}/{iters}]  ACC_Y = {ay * 100:.2f}{cstr}")

    return acc_y, (acc_c or None)


def run_gl_cell(trees, iters: int, trainable: bool = False, epochs: int = 40,
                batch_size: int = 256, lam: float = 1.0, seed: int = 0,
                num_workers: int = 6, train_subset: int = N_TRAIN_SUBSET,
                freeze_backbone: bool = False):
    """OCBM GL-tree CelebA cell: JSON smile tree -> GLTreeCBM (ResNet) -> ACC_Y/ACC_C."""
    import _gltree
    device = _bench.get_device()
    spec = build_spec()
    splits = _load_index()
    tr_files, tr_c, tr_y = splits[0]
    te_files, te_c, te_y = splits[2]
    print(f"OCBM GL ({'ft' if trainable else 'fixed'}) / CelebA   iters={iters}  "
          f"epochs={epochs}  train_subset={train_subset}  device={device}")
    acc_y, acc_c = [], []
    for it in range(iters):
        _bench.set_seed(seed + it)
        perm = torch.randperm(len(tr_files))[:train_subset]
        sub_files = [tr_files[i] for i in perm.tolist()]
        tl = DataLoader(_CelebImages(sub_files, tr_c[perm], tr_y[perm], True),
                        batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
        vl = DataLoader(_CelebImages(te_files, te_c, te_y, False),
                        batch_size=256, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
        model = _gltree.GLTreeCBM(spec.concept_names, trees, spec=spec, feat_dim=512,
                                  backbone=_make_resnet(), trainable=trainable)
        _train(model, tl, device, epochs, lam, True, freeze_backbone=freeze_backbone)
        full, _cpath, cacc = _evaluate(model, vl, device, spec, False)
        acc_y.append(full)
        acc_c.append(cacc)
        print(f"  [iter {it + 1}/{iters}]  ACC_Y = {full * 100:.2f}  ACC_C = {cacc * 100:.2f}")
    return acc_y, acc_c
