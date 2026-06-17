from collections import Counter

import numpy as np
import torch
from loguru import logger
from scipy import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


awa2_groups = {
    'color': [c.lower() for c in 'BLACK, WHITE, BLUE, BROWN, GRAY, ORANGE, RED, YELLOW'.split(', ')],
    'pattern': [c.lower() for c in 'PATCHES, SPOTS, STRIPES'.split(', ')],
    'hair': [c.lower() for c in 'FURRY, HAIRLESS'.split(', ')],
    'skin': ['TOUGHSKIN'.lower()],
    'size': [c.lower() for c in 'BIG, SMALL'.split(', ')],
    'fat':  [c.lower() for c in 'BULBOUS, LEAN'.split(', ')],
    'hand': [c.lower() for c in 'FLIPPERS, HANDS, HOOVES, PADS, PAWS'.split(', ')],
    'leg': ['LONGLEG'.lower()],
    'neck': ['LONGNECK'.lower()],
    'tail': ['TAIL'.lower()],
    'horns': ['HORNS'.lower()],
    'claws': ['CLAWS'.lower()],
    'tusk': ['TUSKS'.lower()],
    'walk': [c.lower() for c in 'BIPEDAL, QUADRAPEDAL'.split(', ')],
    'live': [c.lower() for c in 'ARCTIC, COASTAL, DESERT, BUSH, PLAINS, FOREST, FIELDS, JUNGLE, MOUNTAINS, OCEAN, GROUND, WATER, TREE, CAVE'.split(', ')],
}

concept_groups = dict()


def setup(dataset_name):
    logger.info(f'Loading data for {dataset_name}')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = f'../data/xlsa17/data/{dataset_name}'
    data = io.loadmat(f'{DATA_DIR}/res101.mat')
    attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
    feats = data['features'].T.astype(np.float32)
    labels = data['labels'].squeeze() - 1  # Using "-1" here and for idx to normalize to 0-index
    train_idx = attrs_mat['trainval_loc'].squeeze() - 1
    test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
    test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1
    test_idx = np.array(test_seen_idx.tolist() + test_unseen_idx.tolist())

    logger.info(f'<=============== Preprocessing ===============>')

    attrs = attrs_mat['att'].T
    attrs = torch.from_numpy(attrs).to(DEVICE).float()
    attrs = (attrs > 0).int()

    if dataset_name.lower() == 'awa2':
        all_attributes = {attr for attrs in awa2_groups.values() for attr in attrs}
        with open(f'../data/AwA2-data/Animals_with_Attributes2/predicates.txt', 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
        name2idx = {line[1].lower(): int(line[0]) - 1 for line in lines}
        idx2name = {v: k for k, v in name2idx.items()}
        allowed_idx = np.array([idx for name, idx in name2idx.items() if name in all_attributes])
        attrs = attrs[:, allowed_idx]
        for group in awa2_groups.keys():
            concept_groups[group] = [i for i in range(len(allowed_idx))
                                     if idx2name[allowed_idx[i]] in awa2_groups[group]]
    logger.info(f"Number of used concepts: {attrs.shape[1]}")
    logger.info(f"Concept groups: {concept_groups}")
    return [(feats[i],
             attrs[labels[i]],
             labels[i]) for i in list(train_idx) + list(test_idx)]


class EmbeddingsDataset(Dataset):

    def __init__(self, data, decoy_p):
        self.num_concepts = len(data[0][1]) + (1 if decoy_p > 0 else 0)
        self.data = data
        self.decoy_p = decoy_p

        self.marg_y = Counter([y for x, c, y in self.data])
        _norm = sum(self.marg_y.values())
        for k in self.marg_y.keys():
            self.marg_y[k] /= _norm
        self.num_classes = len(self.marg_y)
        concepts = np.array([d[1].cpu() for d in data])
        self.marg_c = [np.mean(concepts[:, c]) for c in range(self.num_concepts)]
        self.concept_groups = concept_groups
        if decoy_p > 0:
            array = np.zeros(len(data))
            ones_count = int(decoy_p * len(data))
            ones_indices = np.random.choice(len(data), ones_count, replace=False)
            array[ones_indices] = 1
            np.random.shuffle(array)
            self.decoy_concepts = array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, C, y = self.data[idx]
        if self.decoy_p > 0:
            C = torch.cat([C, torch.tensor([self.decoy_concepts[idx]]).to(C.device)]).float()
        return torch.tensor(X), torch.tensor(C), torch.tensor([y]).long()


class EmbeddingsDataModule:

    def __init__(self, dataset_name, decoy_p=0.0, seed=0):
        all_data = setup(dataset_name)
        train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=seed)
                                                 # stratify=[y.item() for x, c, y in all_data])
        train_data, test_data = train_test_split(train_data, test_size=0.4, random_state=seed)
                                                # stratify=[y for X, C, y in train_data])
        logger.info(f"Train size: {len(train_data)} Val size: {len(val_data)} Test size: {len(test_data)}")
        self.train_dataset = EmbeddingsDataset(train_data, decoy_p)
        self.val_dataset = EmbeddingsDataset(val_data, decoy_p)
        self.test_dataset = EmbeddingsDataset(test_data, decoy_p)

    def train_dataloader(self, batch_size=256):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def val_dataloader(self, batch_size=256):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

    def test_dataloader(self, batch_size=1024):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

# if __name__ == '__main__':
#     x = EmbeddingsDataModule(decoy_p=0.3)
#     for x, c, y in x.train_dataloader():
#         print(x.shape, c.shape, y.shape)
#         break
