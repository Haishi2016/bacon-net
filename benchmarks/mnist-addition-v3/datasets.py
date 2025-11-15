# mnist-addition-v3/datasets.py

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


class MNISTDigits(torch.utils.data.Dataset):
    """
    Simple wrapper around torchvision MNIST with fixed normalization.
    Exposes (image, digit_label).
    """

    def __init__(self, root: str, train: bool, download: bool = True):
        self.base = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=mnist_transform(),
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.base[idx]  # (img, digit)


class MNISTAdditionPairs(Dataset):
    """
    Dataset of MNIST digit pairs for the addition task.

    - Underlying MNIST train/test digits.
    - We pre-sample a fixed list of index pairs with a fixed RNG seed so that
      results are reproducible.
    - Each item: ((x1, x2), sum_label) where sum_label ∈ {0, ..., 18}.
    """

    def __init__(
        self,
        root: str,
        train: bool,
        n_pairs: int = 30000,
        seed: int = 0,
        download: bool = True,
    ):
        super().__init__()
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=mnist_transform(),
        )
        self.n_pairs = n_pairs

        rng = np.random.RandomState(seed)
        n = len(self.mnist)
        self.pairs = []

        for _ in range(n_pairs):
            i1 = rng.randint(0, n)
            i2 = rng.randint(0, n)
            # We only care about labels for sum here
            _, d1 = self.mnist[i1]
            _, d2 = self.mnist[i2]
            self.pairs.append((i1, i2, int(d1 + d2)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        i1, i2, s = self.pairs[idx]
        x1, _ = self.mnist[i1]
        x2, _ = self.mnist[i2]
        return (x1, x2), s
