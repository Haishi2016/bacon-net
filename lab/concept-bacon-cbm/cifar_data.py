r"""CIFAR-100 concept data module for the BACON graded-logic CBM (LogicCBM-comparable).

Mirrors ``awa_data`` / ``_cub``: torchvision CIFAR-100 images (upscaled 32->299)
with CLASS-level binary concepts from the MuCIL/LogicCBM concept dictionary
``cifar100_filtered_new.json`` (100 classes -> 925 unique text concepts). Each
image's concept target is its class's 925-d binary concept vector.

Data:
    C:\\School\\datasets\\cifar100\\cifar-100-python\\   (torchvision CIFAR100)
    C:\\School\\datasets\\cifar100\\cifar100_filtered_new.json
"""
from __future__ import annotations

import json
import os

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

CIFAR_ROOT = os.environ.get("CIFAR100_ROOT", r"C:\School\datasets\cifar100")
_CONCEPT_JSON = os.path.join(CIFAR_ROOT, "cifar100_filtered_new.json")

_N_CLASSES = 100


# --------------------------------------------------------------------------- io
def _load_json():
    with open(_CONCEPT_JSON, encoding="utf-8") as f:
        return json.load(f)


def _class_names():
    ds = torchvision.datasets.CIFAR100(root=CIFAR_ROOT, train=False, download=False)
    return ds.classes                                          # label-ordered (0..99)


def build_concept_vocab():
    """Return (concept_names[925], class_concept[100,925]) mapping each CIFAR
    class (torchvision label order) to its concept set. Class names are matched
    by normalizing torchvision's underscores to spaces (json uses spaces)."""
    d = _load_json()
    tv = _class_names()
    # deterministic concept vocabulary = sorted union of all concepts.
    vocab = sorted({c for v in d.values() for c in v})
    cid = {c: i for i, c in enumerate(vocab)}
    M = torch.zeros(_N_CLASSES, len(vocab))
    for label, name in enumerate(tv):
        key = name if name in d else name.replace("_", " ")
        if key not in d:
            raise KeyError(f"CIFAR class {name!r} not found in concept json")
        for c in d[key]:
            M[label, cid[c]] = 1.0
    return vocab, M


def load_concept_names():
    return build_concept_vocab()[0]


def load_class_attr():
    return build_concept_vocab()[1]


# ------------------------------------------------------------------ transforms
def _train_tf(size=299):
    return transforms.Compose([
        transforms.Resize((size + 9, size + 9)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _test_tf(size=299):
    return transforms.Compose([
        transforms.Resize((size + 9, size + 9)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# --------------------------------------------------------------------- dataset
class _CIFARImages(Dataset):
    """CIFAR-100 with class-level 925-d concept targets. Returns
    ``(img(3,299,299), concept(925), label)``."""

    def __init__(self, split="train", train_aug=None, image_size=299):
        if train_aug is None:
            train_aug = (split == "train")
        tf = _train_tf(image_size) if train_aug else _test_tf(image_size)
        self.data = torchvision.datasets.CIFAR100(
            root=CIFAR_ROOT, train=(split == "train"), download=False, transform=tf)
        self.class_attr = load_class_attr()                    # (100, 925)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i]
        return img, self.class_attr[label], label


def find_imbalance():
    """Per-concept imbalance ratio (n_neg/n_pos) over the TRAIN set."""
    M = load_class_attr()                                      # (100, 925)
    ds = torchvision.datasets.CIFAR100(root=CIFAR_ROOT, train=True, download=False)
    counts = torch.zeros(_N_CLASSES)
    for _, label in ds:
        counts[label] += 1
    n_ones = (counts.unsqueeze(1) * M).sum(dim=0)             # (925,)
    total = float(counts.sum())
    return total / n_ones.clamp_min(1.0) - 1.0
