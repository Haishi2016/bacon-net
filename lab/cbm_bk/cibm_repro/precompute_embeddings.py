"""Precompute the InceptionV3 embeddings the official CIBM pipeline trains on.

Faithful to dsb-ifi/cibm ``dataset/CUB.py`` (``embed_image=True``): each image is
resized to 299x299, ImageNet-normalised, and passed once through a pretrained
InceptionV3 with ``fc=Identity`` (eval), caching the 2048-d penultimate feature.
Concepts/labels come from our ``build_cub`` with the canonical 112-concept set
and class-majority labels (matching Koh's pre-processed CBM pickles the paper
uses).

Run from the repo root, e.g.::

    py -3 lab/cbm/cibm_repro/precompute_embeddings.py \
        --root C:\\School\\datasets\\cub\\CUB_200_2011 \
        --out runs/cibm_repro/cub_inception

Produces ``{train,val,test}.pt`` each holding ``{embeds, concepts, labels}``.

NOTE on splits: the official code uses Koh's exact train/val/test pickles
(4796/1198/5794). We don't ship those pickles, so we reproduce the split with a
seeded 20% carve of the official train set (val_fraction=0.2, val_seed=42 ->
4795/1199/5794) -- within one sample of theirs.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

# Make ``cbm_bench`` importable when run as a script.
_LAB_CBM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LAB_CBM not in sys.path:
    sys.path.insert(0, _LAB_CBM)

from cbm_bench.datasets.cub import build_cub  # noqa: E402

# Exactly the official ``basic_transforms`` (dataset/CUB.py): ToTensor ->
# Resize((299,299)) -> ImageNet normalise. Applied to ALL splits (no aug).
_EMBED_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_backbone(device: torch.device) -> torch.nn.Module:
    """Pretrained InceptionV3 with ``fc=Identity`` (the official embedder)."""
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "inception_v3", pretrained=True
    )
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


@torch.no_grad()
def _embed_split(samples, backbone, device, batch_size: int):
    """Embed a list of ``(path, concept_vec, label)`` samples -> cache dict."""
    embeds, concepts, labels = [], [], []
    batch_imgs, batch_concepts, batch_labels = [], [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        feats = backbone(x).cpu()
        embeds.append(feats)
        concepts.append(torch.stack(batch_concepts, dim=0))
        labels.append(torch.tensor(batch_labels, dtype=torch.long))
        batch_imgs.clear()
        batch_concepts.clear()
        batch_labels.clear()

    for path, concept_vec, label in tqdm(samples, desc="Embedding"):
        with Image.open(path) as img:
            tensor = _EMBED_TRANSFORM(img.convert("RGB"))
        batch_imgs.append(tensor)
        batch_concepts.append(torch.as_tensor(concept_vec, dtype=torch.float))
        batch_labels.append(int(label))
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    return {
        "embeds": torch.cat(embeds, dim=0),
        "concepts": torch.cat(concepts, dim=0),
        "labels": torch.cat(labels, dim=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="CUB_200_2011 folder")
    parser.add_argument("--out", required=True, help="output cache directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = {
        "root": args.root,
        "image_size": 299,
        "concept_mode": "class_majority",
        "concept_selection": "canonical112",
        "augmentation": "imagenet",   # irrelevant: we use our own embed transform
        "val_fraction": args.val_fraction,
        "val_seed": args.val_seed,
    }
    bundle = build_cub(cfg)
    splits = {"train": bundle.train, "test": bundle.test}
    if bundle.val is not None:
        splits["val"] = bundle.val

    backbone = _load_backbone(device)
    for name, ds in splits.items():
        print(f"\n=== {name}: {len(ds.samples)} images ===")
        blob = _embed_split(ds.samples, backbone, device, args.batch_size)
        out_path = os.path.join(args.out, f"{name}.pt")
        torch.save(blob, out_path)
        print(
            f"saved {out_path}  embeds={tuple(blob['embeds'].shape)} "
            f"concepts={tuple(blob['concepts'].shape)}"
        )

    print("\nDone. Train with lab/cbm/cibm_repro/train.py --cache", args.out)


if __name__ == "__main__":
    main()
