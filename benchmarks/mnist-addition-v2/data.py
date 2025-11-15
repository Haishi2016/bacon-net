# mnist-addition-2/data.py

import random
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from config import DATASET_CFG, TRAIN_CFG


class MnistAdditionPairs(Dataset):
    """
    Fixed MNIST-addition dataset:
      Each item: ((img1, img2), sum) where sum in {0..18}.

    We pre-sample `n_pairs` (i1, i2, s) indices deterministically using `seed`.
    """
    def __init__(
        self,
        root: str,
        train: bool,
        n_pairs: int,
        seed: int = 123,
        augment: bool = False,
    ):
        super().__init__()

        rng = random.Random(seed)

        # Base transforms: ToTensor + Normalize
        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]

        if augment and train:
            # Keep augmentation *simple* for now; you can tweak later
            aug = [transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))]
            t = transforms.Compose(aug + base_transform)
        else:
            t = transforms.Compose(base_transform)

        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=t,
        )

        n = len(self.mnist)
        self.pairs = []

        for _ in range(n_pairs):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            _, d1 = self.mnist[i1]
            _, d2 = self.mnist[i2]
            self.pairs.append((i1, i2, int(d1 + d2)))  # sum in [0,18]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        i1, i2, s = self.pairs[idx]
        x1, _ = self.mnist[i1]
        x2, _ = self.mnist[i2]
        return (x1, x2), s


def make_loaders(
    dataset_cfg=DATASET_CFG,
    train_cfg=TRAIN_CFG,
    augment_train: bool = False,
):
    """
    Factory to create train / test DataLoaders with shared config.

    Returns:
      train_loader, test_loader
    """
    # Train set
    train_ds = MnistAdditionPairs(
        root=dataset_cfg.data_root,
        train=dataset_cfg.mnist_train,
        n_pairs=dataset_cfg.train_pairs,
        seed=dataset_cfg.train_seed,
        augment=augment_train,
    )

    # Test set – usually use MNIST test split for eval
    test_ds = MnistAdditionPairs(
        root=dataset_cfg.data_root,
        train=dataset_cfg.mnist_test,
        n_pairs=dataset_cfg.test_pairs,
        seed=dataset_cfg.test_seed,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )

    return train_loader, test_loader
