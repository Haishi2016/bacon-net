"""Cached-embedding dataset matching the official ``CUBDataset`` (embed_image=True).

The paper's default (``train_backbone=False``) trains the MLP on precomputed
2048-d InceptionV3 features. This module serves those cached features with the
exact tuple/attribute interface the training loop and MI/HC estimators expect:

* ``dataset[i] -> (embedding (2048,), concepts (num_concepts,), label (1,))``
* ``dataset.num_concepts`` / ``dataset.num_classes``
* ``dataset.marg_y`` (label -> prior) / ``dataset.marg_c`` (per-concept mean)

Build the cache first with ``precompute_embeddings.py``.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingsDataset(Dataset):
    """Serves cached (embedding, concepts, label) triples for one split."""

    def __init__(self, embeds: torch.Tensor, concepts: torch.Tensor, labels: torch.Tensor):
        self.embeds = embeds.float()
        self.concepts = concepts.float()
        self.labels = labels.long().reshape(-1)

        self.num_concepts = int(self.concepts.shape[1])
        self.marg_y = Counter(self.labels.tolist())
        _norm = sum(self.marg_y.values())
        for k in list(self.marg_y.keys()):
            self.marg_y[k] /= _norm
        self.num_classes = len(self.marg_y)
        self.marg_c = [float(self.concepts[:, c].mean()) for c in range(self.num_concepts)]
        # No intervention groups in this minimal reproduction.
        self.concept_groups = {}

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.embeds[idx],
            self.concepts[idx],
            self.labels[idx].reshape(1),
        )


class CUBEmbeddingsDataModule:
    """Loads the precomputed ``{train,val,test}.pt`` caches into datasets."""

    def __init__(self, cache_dir: str, merge_train_val: bool = False):
        self.cache_dir = cache_dir
        train = self._load("train")
        test = self._load("test")
        val = self._load("val")

        if merge_train_val:
            train = {
                k: torch.cat([train[k], val[k]], dim=0) for k in train
            }
            val = None
        self.merge_train_val = merge_train_val

        self.train_dataset = EmbeddingsDataset(**train)
        self.test_dataset = EmbeddingsDataset(**test)
        self.val_dataset = None if val is None else EmbeddingsDataset(**val)

    def _load(self, split: str) -> dict:
        path = os.path.join(self.cache_dir, f"{split}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Embedding cache '{path}' not found. Run "
                "lab/cbm/cibm_repro/precompute_embeddings.py first."
            )
        blob = torch.load(path, map_location="cpu")
        return {
            "embeds": blob["embeds"],
            "concepts": blob["concepts"],
            "labels": blob["labels"],
        }

    @property
    def embed_dim(self) -> int:
        return int(self.train_dataset.embeds.shape[1])

    def train_dataloader(self, batch_size: int = 128) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self, batch_size: int = 128):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def test_dataloader(self, batch_size: int = 128) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
