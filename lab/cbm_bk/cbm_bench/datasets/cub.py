"""CUB-200-2011 dataset for Concept Bottleneck Models.

Expects the *raw* CUB-200-2011 release extracted to ``root``::

    <root>/
        images/                         # bird images grouped by class folder
        images.txt                      # <img_id> <relative_path>
        image_class_labels.txt          # <img_id> <class_id>
        train_test_split.txt            # <img_id> <is_training_image>
        classes.txt                     # <class_id> <class_name>
        attributes/
            image_attribute_labels.txt  # <img_id> <attr_id> <is_present> <certainty> <time>

The official CUB release ships **312** binary attributes (concepts) and **200**
bird classes. Concept supervision uses the per-image ``is_present`` flag.

``root`` is intentionally a path *outside* the ``lab/cbm`` folder (see config),
so large image data lives in a shared ``datasets/`` directory.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .base import DatasetBundle
from ..registry import register_dataset

# The raw CUB release ships 312 binary attributes and 200 classes. After
# class-level majority voting and frequency filtering the effective number of
# concepts is smaller (~112 with the default ``min_class_count=10``).
RAW_N_CONCEPTS = 312
N_CLASSES = 200

# Canonical 112-concept subset frozen by Koh et al. (2020) and reused verbatim
# by every downstream CBM/CEM/ProbCBM/CIBM paper (it ships inside their
# pre-processed ``class_attr_data_10`` pickles, not re-derived from the raw
# annotations). These are 0-based indices into the 312 CUB attributes. Selecting
# this list -- rather than re-running majority voting + frequency filtering --
# guarantees the *exact* concept identity and count (112) the CIBM paper trains
# on, which is required for a faithful numeric comparison: a from-scratch
# majority vote on the raw files yields a slightly different ~109-concept set.
CANONICAL_112 = (
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101,
    106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152,
    153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194,
    196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236,
    238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277,
    283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311,
)


def _read_id_map(path: str) -> dict[int, str]:
    """Read a ``<id> <value...>`` whitespace file into ``{id: value}``."""
    out: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                out[int(parts[0])] = parts[1]
    return out


def _build_transforms(
    image_size: int, train: bool, augmentation: str = "imagenet"
) -> transforms.Compose:
    """Image transforms for CUB.

    ``augmentation`` selects the preprocessing pipeline:

    * ``"imagenet"`` (default) -- ImageNet mean/std normalisation with
      RandomResizedCrop + flip + jitter (train) and Resize->CenterCrop (eval).
      Keeps existing within-harness experiments reproducible.
    * ``"imagenet_strong"`` -- same ImageNet normalisation and eval pipeline as
      ``"imagenet"``, but a heavier *train* augmentation (wider RandomResizedCrop
      scale 0.5-1.0, stronger ColorJitter incl. hue, and a 25% RandomErasing
      cutout). CUB has only ~24 train images/class, so a ResNet18 memorises the
      train set (train acc ->1.0) long before val saturates; this extra
      augmentation shrinks that generalisation gap. Weight decay can't: raising
      it just underfits (see configs). Use for the paper reproduction.
    * ``"paper"`` -- the canonical CBM/CEM/CIBM pipeline (Koh et al., 2020,
      inherited by Galliamov et al.): ``ColorJitter(brightness=32/255,
      saturation=(0.5, 1.5))`` + ``RandomResizedCrop`` + flip on train, a bare
      ``CenterCrop`` (no pre-resize) on eval, and ``Normalize(mean=0.5,
      std=2.0)`` on both. Required to reproduce the paper's absolute numbers.
    """
    if augmentation == "paper":
        # Exact canonical CBM loader transforms (mean 0.5 / std 2.0, eval is a
        # plain CenterCrop with no preceding Resize).
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2.0, 2.0, 2.0])
        if train:
            return transforms.Compose([
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        return transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm,
        ])

    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        if augmentation == "imagenet_strong":
            # MODERATE train aug to curb ResNet18 memorisation on ~24 img/class
            # without over-regularising. (crop 0.5 + jitter+hue + erasing p=0.25
            # was too strong: peak val DROPPED 0.62->0.57 and convergence didn't
            # finish in 100 ep.) Tuned down: crop 0.6, no hue, light erasing.
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.ToTensor(),
                norm,
                transforms.RandomErasing(p=0.1),
            ])
        # RandomResizedCrop + flip + jitter is the standard CUB-CBM
        # augmentation; it is essential to curb overfitting on ~30 images/class.
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            norm,
        ])
    # Deterministic resize + center crop for eval.
    resize = int(round(image_size * 1.15))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])


def _load_instance_concepts(
    root: str, image_ids
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Load per-image attribute labels from CUB.

    Returns ``(present, visible)`` where each maps ``img_id`` to a ``(312,)``
    float tensor. ``present[i][a]`` is the raw ``is_present`` flag; ``visible``
    is ``0`` when the annotator marked the attribute *not visible*
    (``certainty_id == 1``) and ``1`` otherwise. Not-visible labels carry no
    information about the attribute and are excluded from class-majority
    voting (Koh et al., 2020); counting them as genuine absences deflates the
    vote and drops borderline attributes from the filtered concept set.
    """
    attr_path = os.path.join(
        root, "attributes", "image_attribute_labels.txt"
    )
    present = {i: torch.zeros(RAW_N_CONCEPTS) for i in image_ids}
    visible = {i: torch.zeros(RAW_N_CONCEPTS) for i in image_ids}
    with open(attr_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            # Columns: img_id attr_id is_present certainty_id time.
            # Some rows in the official file have a trailing malformed column;
            # guard on the four fields we need.
            if len(parts) < 4:
                continue
            img_id, attr_id, is_present, certainty = (
                int(parts[0]),
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
            )
            if img_id in present:
                present[img_id][attr_id - 1] = float(is_present)
                # certainty_id == 1 == "not visible".
                visible[img_id][attr_id - 1] = 0.0 if certainty == 1 else 1.0
    return present, visible


def _class_majority(
    present: dict[int, torch.Tensor],
    visible: dict[int, torch.Tensor],
    id_to_class: dict[int, int],
    image_ids: list[int],
) -> torch.Tensor:
    """Per-class certainty-aware majority concept prototypes ``(n_classes, 312)``.

    For each class an attribute is marked present if it is labelled present in
    at least half of the images *where it was visible* (annotator certainty
    != "not visible"). This matches the canonical CUB-CBM denoising of Koh et
    al. (2020) inherited by CEM/ProbCBM and the CIBM paper. Excluding
    not-visible labels (rather than treating them as absences) recovers the
    standard ~112-concept set after the downstream >=10-class filter.

    Computed from the train split only to avoid leaking test information.
    """
    pres = torch.zeros(N_CLASSES, RAW_N_CONCEPTS)
    vis = torch.zeros(N_CLASSES, RAW_N_CONCEPTS)
    for img_id in image_ids:
        c = id_to_class[img_id]
        pres[c] += present[img_id]
        vis[c] += visible[img_id]
    frac = torch.where(vis > 0, pres / vis.clamp(min=1), torch.zeros_like(pres))
    return ((frac >= 0.5) & (vis > 0)).float()


def _keep_mask(prototypes: torch.Tensor, min_class_count: int) -> torch.Tensor:
    """Boolean mask of attributes present in at least ``min_class_count`` classes."""
    freq = prototypes.sum(dim=0)
    return freq >= min_class_count


class CUBDataset(Dataset):
    """CUB-200-2011 returning ``(image, concepts, label)``.

    Sample tuples ``(path, concept_vec, label)`` are precomputed by
    :func:`build_cub` so that train and test share identical concept denoising
    and attribute filtering.
    """

    def __init__(
        self,
        samples: List[Tuple[str, torch.Tensor, int]],
        image_size: int,
        train: bool,
        augmentation: str = "imagenet",
    ):
        super().__init__()
        self.samples = samples
        self.transform = _build_transforms(image_size, train, augmentation)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, concept_vec, label = self.samples[idx]
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return image, concept_vec, label


@register_dataset("cub")
def build_cub(cfg) -> DatasetBundle:
    """Factory used by the training driver. ``cfg`` is the dataset config.

    Config keys
    -----------
    root : str
        Path to the extracted ``CUB_200_2011`` folder.
    image_size : int, default 299
    concept_mode : {"class_majority", "instance"}, default "class_majority"
        ``class_majority`` replaces each image's noisy attributes with its
        class prototype (standard CBM denoising); ``instance`` keeps the raw
        per-image labels.
    concept_selection : {"majority_vote", "canonical112"}, default "majority_vote"
        How the concept subset is chosen. ``majority_vote`` re-derives the kept
        attributes from this dataset via certainty-aware class voting + the
        ``min_class_count`` frequency filter (~109 concepts). ``canonical112``
        instead uses the frozen 112-index list shared by the CBM/CEM/CIBM
        papers, giving an exact match to their concept set (use this to
        reproduce paper numbers). ``min_class_count`` is ignored in this mode.
    min_class_count : int, default 10
        Keep only attributes present in at least this many class prototypes
        (10 -> ~109 concepts). Set to 0 to keep all 312. Ignored when
        ``concept_selection == "canonical112"``.
    augmentation : {"imagenet", "paper"}, default "imagenet"
        Image preprocessing pipeline. ``imagenet`` uses ImageNet mean/std and
        the harness's default crops; ``paper`` reproduces the canonical
        CBM/CEM/CIBM transforms (``Normalize(mean=0.5, std=2.0)``, paper jitter,
        bare ``CenterCrop`` eval). Use ``paper`` to match the paper's numbers.
    val_fraction : float, default 0.0
        If > 0, carve this fraction of the *official train* split into a
        validation set (the paper uses 0.2). The split is a seeded global
        shuffle (``val_seed``); concept prototypes are still computed over the
        full official train set, matching the canonical pre-split majority vote.
        When 0 the bundle has no val set and selection falls back to test.
    val_seed : int, default 42
        Seed for the deterministic train/val shuffle (only used when
        ``val_fraction > 0``).
    """
    root = cfg["root"]
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"CUB root '{root}' not found. See lab/cbm/README.md for "
            "download instructions."
        )
    image_size = int(cfg.get("image_size", 299))
    concept_mode = cfg.get("concept_mode", "class_majority")
    concept_selection = str(cfg.get("concept_selection", "majority_vote")).lower()
    if concept_selection not in ("majority_vote", "canonical112"):
        raise ValueError(
            "concept_selection must be 'majority_vote' or 'canonical112', "
            f"got {concept_selection!r}"
        )
    min_class_count = int(cfg.get("min_class_count", 10))
    augmentation = str(cfg.get("augmentation", "imagenet")).lower()
    if augmentation not in ("imagenet", "imagenet_strong", "paper"):
        raise ValueError(
            "augmentation must be 'imagenet', 'imagenet_strong' or 'paper', "
            f"got {augmentation!r}"
        )
    val_fraction = float(cfg.get("val_fraction", 0.0))
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in [0, 1), got {val_fraction}"
        )
    val_seed = int(cfg.get("val_seed", 42))
    image_dir = os.path.join(root, "images")

    id_to_path = _read_id_map(os.path.join(root, "images.txt"))
    id_to_class = {
        i: int(c) - 1  # to 0-based
        for i, c in _read_id_map(
            os.path.join(root, "image_class_labels.txt")
        ).items()
    }
    id_to_split = {
        i: int(s)
        for i, s in _read_id_map(
            os.path.join(root, "train_test_split.txt")
        ).items()
    }

    instance_concepts, instance_visible = _load_instance_concepts(
        root, id_to_path.keys()
    )
    train_ids = [i for i in id_to_path if id_to_split.get(i) == 1]

    # Class prototypes (certainty-aware majority over train) drive concept
    # denoising. Excluding not-visible labels matches the canonical CBM voting.
    prototypes = _class_majority(
        instance_concepts, instance_visible, id_to_class, train_ids
    )
    # Attribute subset. ``canonical112`` pins the frozen 112-index list the CBM/
    # CIBM papers train on (exact reproduction); ``majority_vote`` re-derives it
    # via the ``min_class_count`` frequency filter on the prototypes (~109).
    if concept_selection == "canonical112":
        keep = torch.zeros(RAW_N_CONCEPTS, dtype=torch.bool)
        keep[list(CANONICAL_112)] = True
    elif min_class_count > 0:
        keep = _keep_mask(prototypes, min_class_count)
    else:
        keep = torch.ones(RAW_N_CONCEPTS, dtype=torch.bool)
    n_concepts = int(keep.sum().item())
    prototypes = prototypes[:, keep]

    def concept_for(img_id: int) -> torch.Tensor:
        if concept_mode == "instance":
            return instance_concepts[img_id][keep]
        return prototypes[id_to_class[img_id]]

    def make_sample(img_id: int) -> Tuple[str, torch.Tensor, int]:
        return (
            os.path.join(image_dir, id_to_path[img_id]),
            concept_for(img_id),
            id_to_class[img_id],
        )

    # Partition official-train ids into train/val (paper: 20% val). The split is
    # a deterministic global shuffle so it is reproducible across runs and
    # independent of the model training seed. Prototypes above were computed
    # over the full official train set, matching the canonical pre-split vote.
    official_train = [i for i in id_to_path if id_to_split.get(i) == 1]
    test_ids = [i for i in id_to_path if id_to_split.get(i) == 0]
    if val_fraction > 0.0:
        order = list(official_train)
        gen = torch.Generator().manual_seed(val_seed)
        perm = torch.randperm(len(order), generator=gen).tolist()
        n_val = int(round(val_fraction * len(order)))
        val_pos = set(perm[:n_val])
        train_split_ids = [order[p] for p in range(len(order)) if p not in val_pos]
        val_ids = [order[p] for p in range(len(order)) if p in val_pos]
    else:
        train_split_ids = official_train
        val_ids = []

    def make_dataset(ids: list[int], is_train: bool) -> CUBDataset:
        samples = [make_sample(i) for i in ids]
        if not samples:
            raise RuntimeError(
                f"No CUB samples for split is_train={is_train} under '{root}'."
            )
        return CUBDataset(samples, image_size, train=is_train, augmentation=augmentation)

    class_names = None
    classes_file = os.path.join(root, "classes.txt")
    if os.path.isfile(classes_file):
        class_map = _read_id_map(classes_file)
        class_names = [class_map[i + 1] for i in range(N_CLASSES)]

    val_set = make_dataset(val_ids, is_train=False) if val_ids else None
    return DatasetBundle(
        name="cub",
        train=make_dataset(train_split_ids, is_train=True),
        test=make_dataset(test_ids, is_train=False),
        n_concepts=n_concepts,
        n_classes=N_CLASSES,
        val=val_set,
        class_names=class_names,
    )
