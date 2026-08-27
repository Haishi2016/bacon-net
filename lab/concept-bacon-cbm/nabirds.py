"""NABirds loader + CUB->NABirds species mapping for cross-dataset faithfulness.

NABirds (Van Horn et al., 2015) is a separate ~48k-image North-American bird
dataset with 555 visual-category leaf classes under a ~1011-node hierarchy.  It
does NOT ship the CUB 112-attribute matrix, so we transfer CUB's *species-level*
(class-majority) attribute vectors onto NABirds images whose species matches a
CUB species.  A concept that means "striped wing" on CUB should still fire on
NABirds birds whose CUB-mapped species is striped -- that preservation is the
faithfulness signal (parallels the MNIST->USPS test).

Expected NABirds layout under --nabirds ROOT (standard release):
    ROOT/images/<class>/<image>.jpg
    ROOT/images.txt              <image_id> <relpath>
    ROOT/image_class_labels.txt  <image_id> <class_id>
    ROOT/classes.txt             <class_id> <class name>
    ROOT/hierarchy.txt           <child_id> <parent_id>
    ROOT/train_test_split.txt    <image_id> <is_train>   (optional; we use all)

    import nabirds
    ds = nabirds.NABirdsCUB(ROOT)     # yields (img, cub_attr112, cub_class)
"""

from __future__ import annotations

import os
import re
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "table"))

import _cub                                                     # noqa: E402


# ---------------------------------------------------------------- name matching
def _tokens(s: str) -> frozenset:
    """Lower-cased alpha tokens, singularised (drop trailing 's') for matching.

    Handles CUB "Black_footed_Albatross" vs NABirds "Black-footed Albatross"
    and possessives ("Brewer's" -> "brewer" == CUB "Brewer")."""
    toks = re.findall(r"[a-z]+", s.lower())
    out = set()
    for t in toks:
        if len(t) < 2 or t in ("the", "of", "and"):    # drop possessive 's', stopwords
            continue
        out.add(t[:-1] if len(t) > 3 and t.endswith("s") else t)
    return frozenset(out)


def _read_id_map(path: str) -> dict:
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def load_index(root: str):
    """Return (images{id:relpath}, labels{id:class_id}, classes{cid:name},
    parent{child:parent})."""
    images = _read_id_map(os.path.join(root, "images.txt"))
    labels = _read_id_map(os.path.join(root, "image_class_labels.txt"))
    classes = _read_id_map(os.path.join(root, "classes.txt"))
    parent = {}
    hpath = os.path.join(root, "hierarchy.txt")
    if os.path.exists(hpath):
        with open(hpath, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 2:
                    parent[p[0]] = p[1]
    return images, labels, classes, parent


def _ancestors(cid: str, parent: dict):
    chain, seen = [cid], {cid}
    while cid in parent and parent[cid] not in seen:
        cid = parent[cid]
        chain.append(cid)
        seen.add(cid)
    return chain


# ---------------------------------------------------------------- CUB side
def cub_class_names():
    """200 CUB species names (species part only, e.g. 'Black_footed_Albatross')."""
    names = [None] * 200
    with open(os.path.join(_cub.CUB, "classes.txt"), "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(" ", 1)
            if len(p) == 2:
                idx = int(p[0]) - 1
                names[idx] = p[1].split(".", 1)[1]              # drop "001."
    return names


def cub_class_attr():
    """Map CUB class_label (0-based) -> 112-dim class-majority attribute vector.

    The Koh pkls already store class-majority attribute labels, so any image of
    a class carries that class's vector."""
    table = {}
    ds = _cub._CUBImages("train", train_aug=False)
    for e in ds.entries:
        y = e["class_label"]
        if y not in table:
            table[y] = torch.tensor(e["attribute_label"], dtype=torch.float32)
        if len(table) == 200:
            break
    return table


# ---------------------------------------------------------------- mapping
def build_mapping(root: str, verbose: bool = True):
    """Return (samples, class_attr) where samples = list of (relpath, cub_class)
    for every NABirds image whose species maps to a CUB species."""
    images, labels, classes, parent = load_index(root)
    names = cub_class_names()
    class_attr = cub_class_attr()
    cub_tok = {}
    for i, n in enumerate(names):
        cub_tok[_tokens(n.replace("_", " "))] = i               # token-set -> cub idx

    # NABirds class_id -> cub class (match self or nearest ancestor by token-set)
    na2cub = {}
    for cid in classes:
        for anc in _ancestors(cid, parent):
            key = _tokens(classes[anc])
            if key in cub_tok:
                na2cub[cid] = cub_tok[key]
                break

    samples = []
    for img_id, relpath in images.items():
        cid = labels.get(img_id)
        if cid is not None and cid in na2cub:
            samples.append((relpath, na2cub[cid]))

    if verbose:
        matched = len({v for v in na2cub.values()})
        print(f"NABirds mapping: {len(na2cub)}/{len(classes)} NA classes -> "
              f"{matched}/200 CUB species; {len(samples)} images usable.")
    return samples, class_attr


# ---------------------------------------------------------------- dataset
class NABirdsCUB(Dataset):
    """NABirds images with CUB class-majority 112-attr targets (mapped species).

    Yields (img_tensor, attr112, cub_class) mirroring _cub._CUBImages so the
    same `collect(...)` works unchanged."""

    def __init__(self, root: str, verbose: bool = True):
        self.root = root
        self.samples, self.class_attr = build_mapping(root, verbose=verbose)
        self.tf = _cub._TEST_TF
        self.img_dir = os.path.join(root, "images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        relpath, y = self.samples[i]
        img = Image.open(os.path.join(self.img_dir, relpath)).convert("RGB")
        return self.tf(img), self.class_attr[y], y


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else r"C:\School\datasets\nabirds"
    ds = NABirdsCUB(root)
    print(f"NABirdsCUB: {len(ds)} images across "
          f"{len({y for _, y in ds.samples})} mapped CUB species.")
    x, a, y = ds[0]
    print("sample:", x.shape, "attr sum", int(a.sum().item()), "cub_class", y)
