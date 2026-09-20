"""Diagnostic: which input pipeline makes the pretrained InceptionV3 features
most class-separable on CUB? Frozen features -> linear probe -> class acc.

Variants:
  A: ImageNet norm  + transform_input=False   (correct modern torchvision usage)
  B: Koh norm 0.5/2 + transform_input=False   (what we were doing -- suspected bug)
  C: Koh norm 0.5/2 + transform_input=True     (Koh's exact legacy pipeline)
"""
from __future__ import annotations
import os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
import _cub                                                    # noqa: E402

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
KOH = ([0.5, 0.5, 0.5], [2.0, 2.0, 2.0])


def _tf(mean, std, train):
    if train:
        return transforms.Compose([transforms.RandomResizedCrop(299),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])


class _DS(torch.utils.data.Dataset):
    def __init__(self, split, tf):
        import pickle
        self.e = pickle.load(open(os.path.join(_cub.CUB, f"{split}.pkl"), "rb"))
        self.tf = tf
    def __len__(self): return len(self.e)
    def __getitem__(self, i):
        from PIL import Image
        e = self.e[i]
        img = Image.open(_cub._local_path(e["img_path"])).convert("RGB")
        return self.tf(img), e["class_label"]


def _backbone(transform_input, device):
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                              transform_input=transform_input, aux_logits=True)
    net.fc = nn.Identity(); net.AuxLogits = None; net.aux_logits = False
    return net.to(device).eval()


@torch.no_grad()
def _feats(net, mean, std, split, device):
    dl = DataLoader(_DS(split, _tf(mean, std, False)), batch_size=64, shuffle=False,
                    num_workers=10, pin_memory=True)
    X, Y = [], []
    for img, y in dl:
        X.append(net(img.to(device)).cpu()); Y.append(y)
    return torch.cat(X), torch.cat(Y)


def probe(name, mean, std, transform_input, device):
    net = _backbone(transform_input, device)
    Xtr, Ytr = _feats(net, mean, std, "train", device)
    Xte, Yte = _feats(net, mean, std, "test", device)
    # standardize features, then linear probe
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    clf = nn.Linear(Xtr.size(1), 200).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Xte, Yte = Xtr.to(device), Ytr.to(device), Xte.to(device), Yte.to(device)
    for ep in range(300):
        opt.zero_grad(); F.cross_entropy(clf(Xtr), Ytr).backward(); opt.step()
    acc = (clf(Xte).argmax(1) == Yte).float().mean().item()
    print(f"  {name}: frozen-feature linear probe test acc = {acc*100:.2f}%", flush=True)
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== InceptionV3 frozen-feature separability by input pipeline =====", flush=True)
    probe("A ImageNet-norm  + transform_input=False", *IMAGENET, False, device)
    probe("B Koh-norm 0.5/2 + transform_input=False (ours)", *KOH, False, device)
    probe("C Koh-norm 0.5/2 + transform_input=True  (Koh)", *KOH, True, device)


if __name__ == "__main__":
    main()
