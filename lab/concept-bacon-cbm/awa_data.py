r"""AwA2 (Animals with Attributes 2) data module for the BACON graded-logic CBM.

Mirrors the CUB ``_cub`` interface so the same OCBM pipeline (joint concept
pretrain -> recttree GL tree head) runs on AwA2 with only ``(K, n_classes,
concept source)`` swapped -- staying comparable with LogicCBM (Vemuri et al.,
WACV 2026), which trains end-to-end InceptionV3 on the **85 binary predicates**
(class-level) over the 50 AwA2 classes.

Layout (official cvml.ista.ac.at/AwA2 release, unzipped):
    Animals_with_Attributes2/
        JPEGImages/<class_name>/<class_name>_NNNN.jpg
        classes.txt                  (idx <tab> class_name)
        predicates.txt               (idx <tab> predicate_name)
        predicate-matrix-binary.txt  (50 x 85 class-level binary attributes)

Split: LogicCBM classifies all 50 classes, so we use a SEEDED per-class
train/test split (not the zero-shot seen/unseen split), default 80/20.
"""
from __future__ import annotations

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

AWA_ROOT = os.environ.get(
    "AWA2_ROOT", r"C:\School\datasets\awa2\Animals_with_Attributes2")

_N_CLASSES = 50
_N_ATTR = 85


# --------------------------------------------------------------------------- io
def _root(root=None):
    return root or AWA_ROOT


def load_class_names(root=None):
    """50 AwA2 class names in file order (index = class label)."""
    names = []
    with open(os.path.join(_root(root), "classes.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                names.append(p[1].strip())
    return names


def load_attr_names(root=None):
    """85 AwA2 predicate (attribute) names in file order."""
    names = []
    with open(os.path.join(_root(root), "predicates.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                names.append(p[1].strip())
    return names


def load_class_attr(root=None):
    """(50, 85) class-level binary attribute matrix (predicate-matrix-binary)."""
    rows = []
    with open(os.path.join(_root(root), "predicate-matrix-binary.txt"),
              encoding="utf-8") as f:
        for line in f:
            vals = line.split()
            if vals:
                rows.append([int(v) for v in vals])
    M = torch.tensor(rows, dtype=torch.float32)
    if M.shape != (_N_CLASSES, _N_ATTR):
        raise ValueError(f"expected ({_N_CLASSES},{_N_ATTR}) predicate matrix, got {tuple(M.shape)}")
    return M


# ------------------------------------------------------------------ transforms
def _train_tf(size=299):
    return transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _test_tf(size=299):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ------------------------------------------------------------------- listing
def _list_images(root=None):
    """Return sorted list of (img_path, class_idx) over JPEGImages/<class>/*.jpg."""
    root = _root(root)
    names = load_class_names(root)
    cls_to_idx = {n: i for i, n in enumerate(names)}
    img_dir = os.path.join(root, "JPEGImages")
    items = []
    for cname in names:
        cdir = os.path.join(img_dir, cname)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(cdir, fn), cls_to_idx[cname]))
    if not items:
        raise RuntimeError(f"no AwA2 images found under {img_dir}")
    return items


def _split_indices(items, split, seed=42, test_frac=0.2):
    """Seeded per-class train/test split (all 50 classes in both)."""
    import random
    by_cls = {}
    for i, (_, c) in enumerate(items):
        by_cls.setdefault(c, []).append(i)
    rng = random.Random(seed)
    keep = []
    for c, idxs in by_cls.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_frac)))
        test_ids = set(idxs[:n_test])
        for i in idxs:
            in_test = i in test_ids
            if (split == "test") == in_test:
                keep.append(i)
    return keep


# --------------------------------------------------------------------- dataset
class _AwAImages(Dataset):
    """AwA2 image dataset. Returns ``(img, class_attr_vec(85), class_label)`` --
    concept target is the image's CLASS-level 85-d binary predicate vector,
    matching the CUB ``attr312`` (class-level) supervision."""

    def __init__(self, split="train", train_aug=None, image_size=299,
                 seed=42, test_frac=0.2, root=None):
        self.root = _root(root)
        if train_aug is None:
            train_aug = (split == "train")
        self.tf = _train_tf(image_size) if train_aug else _test_tf(image_size)
        items = _list_images(self.root)
        keep = _split_indices(items, split, seed=seed, test_frac=test_frac)
        self.items = [items[i] for i in keep]
        self.class_attr = load_class_attr(self.root)          # (50, 85)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        c = self.class_attr[label]
        return self.tf(img), c, label


def find_imbalance(seed=42, test_frac=0.2, root=None):
    """Per-attribute imbalance ratio (n_neg/n_pos) over the TRAIN images, matching
    Koh/LogicCBM ``find_class_imbalance`` -> scalar weight per attribute BCE."""
    root = _root(root)
    M = load_class_attr(root)                                  # (50, 85)
    items = _list_images(root)
    keep = _split_indices(items, "train", seed=seed, test_frac=test_frac)
    counts = torch.zeros(_N_CLASSES)
    for i in keep:
        counts[items[i][1]] += 1
    n_ones = (counts.unsqueeze(1) * M).sum(dim=0)             # (85,) positives
    total = float(counts.sum())
    return total / n_ones.clamp_min(1.0) - 1.0               # (85,)
