# mnist-addition-2/config.py

from dataclasses import dataclass

@dataclass
class DatasetConfig:
    data_root: str = "./data"
    train_pairs: int = 30000    # canonical size (matches many papers)
    test_pairs: int = 10000
    mnist_train: bool = True    # use MNIST train split for pairs
    mnist_test: bool = False    # use MNIST test split for eval pairs
    train_seed: int = 123
    test_seed: int = 999

@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    seed: int = 0
    num_workers: int = 2
    pin_memory: bool = True

@dataclass
class BackboneConfig:
    feat_dim: int = 128
    use_batchnorm: bool = True   # if you later want a "strict" baseline, set False

# One “global” config object you can import everywhere
DATASET_CFG = DatasetConfig()
TRAIN_CFG = TrainConfig()
BACKBONE_CFG = BackboneConfig()
