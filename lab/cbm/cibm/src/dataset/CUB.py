import gc
import os.path as osp
import pickle
from copy import copy
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

CUB_DATA_PATH = 'C:/School/datasets/cub/CUB_200_2011/'
original_concept_ids = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
                        51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101,
                        106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152,
                        153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194,
                        196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236,
                        238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277,
                        283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

concept_groups = {}


def setup(n_removed, n_decoys):
    global concept_groups
    dfs = []
    for filename in ['train.pkl', 'val.pkl', 'test.pkl']:
        with open(osp.join(CUB_DATA_PATH, filename), 'rb') as f:
            df = pickle.load(f)
            df = pd.DataFrame(df)
        df.rename(columns={
            'attribute_label': 'is_present',
            'img_path': 'image_path',
            'class_label': 'label'}, inplace=True)
        df['image_path'] = df['image_path'].apply(
            lambda p: CUB_DATA_PATH + p.split('CUB_200_2011')[-1]
        )
        dfs.append(df)
    with open(osp.join(CUB_DATA_PATH, '../attributes.txt'), 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
        attributes = {int(k): v for k, v in lines if int(k) in original_concept_ids}
    for concept_id, concept_name in attributes.items():
        concept_group_name = concept_name.split('::')[0]
        if concept_group_name not in concept_groups:
            concept_groups[concept_group_name] = []
        concept_groups[concept_group_name].append(original_concept_ids.index(concept_id))
    concepts = dfs[0].is_present.to_list() + dfs[1].is_present.to_list() + dfs[2].is_present.to_list()
    concepts = np.array(concepts)
    if n_removed > 0:
        indices_to_keep = np.sort(np.random.choice(len(concepts[0]),
                                            size=len(concepts[0]) - n_removed, replace=False))
        concepts = concepts[:, indices_to_keep]
        for group_name, indices in concept_groups.items():
            new_indices = [np.where(indices_to_keep == idx)[0][0] for idx in indices if idx in indices_to_keep]
            concept_groups[group_name] = new_indices
    if n_decoys > 0:
        marginal_probs = [np.random.random() / 2 for _ in range(n_decoys)]
        concepts_to_add = np.array([[int(np.random.random() < prob) for prob in marginal_probs ]
                                        for _ in range(len(concepts))])  # (N, n_decoys)
        decoy_indices = [len(concepts[0]) + i for i in range(n_decoys)]
        concepts = np.concatenate([concepts, concepts_to_add], axis=1)

        for idx in decoy_indices:  # scatter around random groups for better simulation
            key = np.random.choice(list(concept_groups.keys()))
            concept_groups[key].append(idx)
    concepts = [concepts[:len(dfs[0]), :],
                concepts[len(dfs[0]):len(dfs[0]) + len(dfs[1]), :], 
                concepts[len(dfs[0]) + len(dfs[1]):, :]]
    for i in range(3):
        new = [list(concepts[i][idx]) for idx in range(len(dfs[i]))]
        dfs[i].is_present = new
    return dfs


class CUBDataset(Dataset):
    def __init__(self,
                 data,
                 image_transforms,
                 embed_image=True):
        super().__init__()
        self.labels = data.label.to_numpy().reshape(-1)
        self.images_paths = data.image_path.to_list()
        self.concepts = np.array(data.is_present.to_list())
        self.concept_groups = copy(concept_groups)

        self.num_concepts = len(self.concepts[0])
        self.marg_y = Counter(self.labels)
        self.marg_c = [np.mean(self.concepts[:, c]) for c in range(self.num_concepts)]
        for k in self.marg_y:
            self.marg_y[k] /= len(self.labels)
        self.num_classes = len(self.marg_y)

        images = []
        for idx in tqdm(range(len(self)), desc='Reading images'):
            with open(self.images_paths[idx], 'rb') as f:
                img = Image.open(f)
                image = np.array(img)
                if len(image.shape) == 2:
                    image = np.stack([image, image, image]).transpose((1, 2, 0))
                img.close()
            if embed_image:
                images.append(image_transforms(image))
            else:
                images.append(image)
        self.embed_image = embed_image
        if self.embed_image:
            self.images_embeds = []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3',
                                   pretrained=True).to(device)
            model.fc = torch.nn.Identity()
            model.eval()
            images_loader = DataLoader(images, batch_size=128, shuffle=False, drop_last=False)

            for img_tensor in tqdm(images_loader,
                                   desc=f'Embeddings images with {model.__class__} backbone'):
                with torch.no_grad():
                    self.images_embeds.append(model(img_tensor.to(device)).cpu())
            del images
            model.cpu()
            del model
            torch.cuda.empty_cache()
            gc.collect()
            self.images_embeds = torch.vstack(self.images_embeds)
        else:
            self.images = images
            self.transforms = image_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.embed_image:
            return self.images_embeds[idx], \
                torch.Tensor(self.concepts[idx]), \
                torch.Tensor([self.labels[idx]]).long()
        else:
            return self.transforms(self.images[idx]), \
                torch.Tensor(self.concepts[idx]), \
                torch.Tensor([self.labels[idx]]).long()


class CUBDataModule:
    def __init__(self, embed_image=True, merge_train_val=False, n_decoys=0, n_removed=0):
        basic_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if not embed_image:
            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Resize((299, 299)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transforms = basic_transforms

        self.train_data, self.val_data, self.test_data = setup(n_decoys=n_decoys, n_removed=n_removed)
        self.merge_train_val = merge_train_val

        if merge_train_val:
            self.train_data = pd.concat([self.train_data, self.val_data]).reset_index(drop=True)
        logger.info(
            f"Training samples: {len(self.train_data)}, testing samples: {len(self.test_data)}")
        logger.info(f"Train and val merged: {self.merge_train_val}")

        self.train_dataset = CUBDataset(data=self.train_data,
                                        image_transforms=train_transforms,
                                        embed_image=embed_image)
        if not self.merge_train_val:
            self.val_dataset = CUBDataset(data=self.val_data,
                                          image_transforms=basic_transforms,
                                          embed_image=embed_image)
        self.test_dataset = CUBDataset(data=self.test_data,
                                       image_transforms=basic_transforms,
                                       embed_image=embed_image)
        logger.info(f"Concept groups: {self.train_dataset.concept_groups}")

    def train_dataloader(self, batch_size=128):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self, batch_size=128):
        if self.merge_train_val:
            return None
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self, batch_size=128):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


if __name__ == '__main__':
    x = CUBDataModule()
    im, c, y = x.train_dataset[0]
    print(im.shape, c.shape, y.shape)
    print(len(x.train_dataset), len(x.test_dataset))
    for i, (im, c, y) in enumerate(x.train_dataloader()):
        if i % 10 == 0:
            print(im.shape, c.shape, y.shape)
