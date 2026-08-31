"""
Shared helpers for the CUB-200-2011 table cells (END-TO-END fine-tuned ResNet-18).

Unlike the frozen-feature cub_compare.py, here the ResNet-18 backbone is
fine-tuned end-to-end (the standard CUB-CBM protocol), which is what reproduces
the paper's ~75/73 numbers.  Reuses the generalized model classes from
cream/models.py (BlackBox / SoftCBM / SoftCBMSC / CREAM / CtrueY now accept an
injectable backbone + derive the class count from spec.A_Y).

Model keys -> paper rows (same as FMNIST):
  blackbox / ctruey / cbm / cbm+sc / cream-wo-sc / cream

CUB concepts = 112 INDEPENDENT binary attributes (no mutex groups); concept
targets are the per-image attribute labels (class-majority in the Koh pkls).
"""

from __future__ import annotations

import os
import pickle
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

import _bench                                                   # sets sys.path
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_bench._CBM, "cub"))          # for cub_trees
from run_cream import concept_accuracy                          # noqa: E402
from models import BlackBox, CtrueY, SoftCBM, SoftCBMSC, CREAM, BaconCBM  # noqa: E402

CUB = r"C:\School\datasets\cub\CUB_200_2011"
_ATTR_TXT = r"C:\School\datasets\cub\attributes.txt"          # 312 attribute names
_ATTR_CONT = _os.path.join(CUB, "attributes", "class_attribute_labels_continuous.txt")
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# 0-based indices of the canonical 112 concepts into the 312 raw attributes.
CANONICAL_112 = (
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101,
    106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152,
    153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194,
    196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236,
    238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277,
    283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311,
)
_SUPERCATS = ["color", "pattern", "shape", "length", "size"]


_CLASS_ATTR_312 = None
_NAMES_312 = None


def load_names_312():
    """All 312 raw CUB attribute names (0-based order, as in attributes.txt)."""
    global _NAMES_312
    if _NAMES_312 is None:
        raw = {}
        with open(_ATTR_TXT, "r", encoding="utf-8") as f:
            for line in f:
                p = line.split(None, 1)
                if len(p) == 2:
                    raw[int(p[0]) - 1] = p[1].strip()
        _NAMES_312 = [raw[i] for i in range(len(raw))]
    return _NAMES_312


def load_class_attr_312(thresh=50.0):
    """(200, 312) binary class-attribute matrix (Koh class-level attributes).

    Built from ``class_attribute_labels_continuous.txt`` (percentage of images
    per class exhibiting each attribute) binarized at ``thresh`` (>=50% => the
    attribute is present for that class). All images of a class share this row --
    the standard CUB-CBM 312 setup the LogicCBM paper uses.
    """
    global _CLASS_ATTR_312
    if _CLASS_ATTR_312 is None:
        rows = []
        with open(_ATTR_CONT, "r", encoding="utf-8") as f:
            for line in f:
                vals = line.split()
                if vals:
                    rows.append([float(v) for v in vals])
        M = torch.tensor(rows, dtype=torch.float32)             # (200, 312) in [0,100]
        _CLASS_ATTR_312 = (M >= thresh).float()
    return _CLASS_ATTR_312


def _load_attr_groups():
    """Group the 112 canonical concepts by attribute TYPE and SUPER-CATEGORY.

    Returns (names112, type_of_concept[K], type_names, supercat_of_type[T]).
    Attribute name format: 'has_<part>_<attr>::<value>' -> type = prefix before '::'.
    """
    raw = {}
    with open(_ATTR_TXT, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split(None, 1)
            if len(p) == 2:
                raw[int(p[0]) - 1] = p[1].strip()               # 0-based id -> name
    names112 = [raw[a] for a in CANONICAL_112]
    types = [n.split("::")[0] for n in names112]                # e.g. has_wing_color
    type_names = sorted(set(types), key=types.index)
    tid = {t: i for i, t in enumerate(type_names)}
    type_of_concept = [tid[t] for t in types]                   # (K,)

    def _supercat(t):
        for kw in _SUPERCATS:
            if kw in t:
                return kw
        return "shape"
    supercat_of_type = [_SUPERCATS.index(_supercat(t)) for t in type_names]  # (T,)
    return names112, type_of_concept, type_names, supercat_of_type


def _norm_membership(child_of_parent, n_parent, device):
    """Uniform membership matrix M (n_parent, n_child): row p = 1/|p| over its children."""
    n_child = len(child_of_parent)
    M = torch.zeros(n_parent, n_child, device=device)
    for c, p in enumerate(child_of_parent):
        M[p, c] = 1.0
    return M / M.sum(dim=1, keepdim=True).clamp_min(1.0)



LABELS = {
    "blackbox": "Black-box", "ctruey": "C_true->Y", "cbm": "CBM",
    "cbm+sc": "CBM+SC", "cream-wo-sc": "CREAM w/o SC", "cream": "CREAM",
    "ocbm": "OCBM", "ocbm-ft": "OCBM (fine-tuned)",
}

_TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
_TEST_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
])


def _local_path(img_path: str) -> str:
    i = img_path.replace("\\", "/").find("images/")
    return os.path.join(CUB, img_path.replace("\\", "/")[i:])


class _CUBImages(Dataset):
    def __init__(self, split: str, train_aug: bool, attr312: bool = False):
        with open(os.path.join(CUB, f"{split}.pkl"), "rb") as f:
            self.entries = pickle.load(f)
        self.tf = _TRAIN_TF if train_aug else _TEST_TF
        # class-level 312 attribute matrix (all images of a class share a row).
        self.attr312 = load_class_attr_312() if attr312 else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = Image.open(_local_path(e["img_path"])).convert("RGB")
        if self.attr312 is not None:
            c = self.attr312[e["class_label"]]
        else:
            c = torch.tensor(e["attribute_label"], dtype=torch.float32)
        return self.tf(img), c, e["class_label"]


def build_spec(ay_mode: str = "positive"):
    """CUB ConceptSpec-like object (112 binary concepts, 200 classes).

    ``ay_mode`` controls the CREAM concept->class connectivity mask (spec.A_Y):
      positive : class reads only its ON prototype attributes (default; matches
                 the class-attribute matrix but can't use discriminative absences).
      signed   : class also reads attributes it confidently LACKS (|2f-1|>=0.5),
                 so absent-attribute evidence can discriminate.
      full     : class reads all concepts (dense C->Y; upper bound = CBM head).
    Concept SUPERVISION and the C-C block still use the positive prototypes.
    """
    from cub_trees import build_prototypes, load_frequencies, load_split
    C, y = load_split("train")
    P = build_prototypes(C, y, 200)                 # (200, 112) in {0,1}
    K = P.shape[1]
    names = [f"c{j}" for j in range(K)]
    spec = types.SimpleNamespace()
    spec.concept_names = names
    spec.K = K
    spec.mutex_groups = []
    spec.binary_concepts = list(range(K))
    spec.index = {n: i for i, n in enumerate(names)}
    if ay_mode == "full":
        spec.A_Y = torch.ones_like(P)
    elif ay_mode == "signed":
        F, _ = load_frequencies(P)                  # (200, 112) class freq in [0,1]
        spec.A_Y = ((2.0 * F - 1.0).abs() >= 0.5).float()   # strong ON or OFF
    else:                                            # positive (default)
        spec.A_Y = P
    spec.Y = P
    spec.class_on = {k: [names[j] for j in range(K) if P[k, j] > 0.5]
                     for k in range(200)}
    # OCBM/BACON: per-class signed-AND tree from the prototype (positive literal
    # where the attribute is ON, NOT literal where OFF). Complete (ceiling 100).
    spec.formulas = {
        k: " AND ".join(names[j] if P[k, j] > 0.5 else f"NOT {names[j]}"
                        for j in range(K))
        for k in range(200)
    }
    spec.concept_targets = lambda labels: P.to(labels.device)[labels]
    return spec


def _make_resnet():
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    return net


class _CUBOCBM(nn.Module):
    """Vectorized CUB BACON-CBM (OCBM): ResNet -> 112 sigmoid concepts -> per-class
    signed graded-AND trees (geometric mean of signed concept match, as in
    cub_trees), computed in one batched op so 200 trees are fast.

    finetune=True keeps the tree STRUCTURE fixed (which concepts, signed how) but
    trains per-(class, concept) input weights -- a concept a class barely depends
    on can be down-weighted, relaxing the strict 112-way conjunction.
    """

    def __init__(self, P, backbone, finetune=False, temp=6.0):
        super().__init__()
        import math
        L, K = P.shape
        self.backbone = backbone
        self.concept = nn.Linear(512, K)
        self.register_buffer("sign", P.float())            # (L, K) prototype sign
        self.finetune = finetune
        self.log_temp = nn.Parameter(torch.tensor(float(math.log(temp))))
        if finetune:
            self.wlogit = nn.Parameter(torch.zeros(L, K))  # per-(class,concept) weight

    def forward(self, x, use_side=True):
        c = torch.sigmoid(self.concept(self.backbone(x)))  # (B, K)
        s = self.sign.unsqueeze(0)                         # (1, L, K)
        cc = c.unsqueeze(1)                                # (B, 1, K)
        match = (s * cc + (1.0 - s) * (1.0 - cc)).clamp_min(1e-6)   # (B, L, K)
        logm = torch.log(match)
        if self.finetune:
            w = torch.softmax(self.wlogit, dim=1).unsqueeze(0)      # (1, L, K)
            truth = torch.exp((logm * w).sum(-1))          # weighted geometric AND
        else:
            truth = torch.exp(logm.mean(-1))               # geometric AND (ceiling 100)
        truth = truth.clamp(1e-6, 1.0 - 1e-6)
        logits = self.log_temp.exp() * (torch.log(truth) - torch.log1p(-truth))
        return logits, c


class _CUBGLOCBM(nn.Module):
    """Prompt-faithful HIERARCHICAL GL CUB OCBM (OCBM_V1_TREE_PROMPT.md applied
    programmatically): ResNet -> 112 sigmoid concepts -> per-species tree that
    aggregates concept -> attribute-TYPE sub-score -> SUPER-CATEGORY -> species,
    each level a weighted graded-AND (geometric mean of signed concept match).
    Signs come from the class prototype.  Vectorized via segment (einsum)
    reductions so all 200 species evaluate at once.

    finetune=True keeps the hierarchy fixed but learns the per-level weights
    (masked softmax within each parent) -- the "input weights" of the GL tree.
    """

    def __init__(self, P, backbone, finetune=False, temp=6.0):
        super().__init__()
        import math
        L, K = P.shape
        self.backbone = backbone
        self.concept = nn.Linear(512, K)
        self.register_buffer("sign", P.float())                 # (L, K)
        _, toc, tnames, sot = _load_attr_groups()
        self.T, self.S = len(tnames), len(_SUPERCATS)
        self.finetune = finetune
        self.log_temp = nn.Parameter(torch.tensor(float(math.log(temp))))
        gmem = _CUBGLOCBM._mem(toc, self.T)                     # (T, K) 0/1
        smem = _CUBGLOCBM._mem(sot, self.S)                     # (S, T) 0/1
        # Importance weights (data-free, per OCBM_V1_TREE_PROMPT principle 3):
        # weight each concept by how much its prototype membership VARIES across
        # species.  A concept that is ON for (almost) all or (almost) no species
        # carries no discriminative signal; one ON for ~half is maximally useful.
        # Propagate this mass up the hierarchy so more discriminative attribute
        # TYPES and SUPER-CATEGORIES get proportionally larger weight, instead of
        # uniform weights that dilute the many highly-varying colour attributes.
        pbar = P.float().mean(0)                                # (K,) ON-fraction
        disc = (pbar * (1.0 - pbar)).clamp_min(1e-4)            # (K,) in (0, .25]
        gw = gmem * disc.unsqueeze(0)                           # (T, K) weighted
        Gmat = gw / gw.sum(1, keepdim=True).clamp_min(1e-6)
        Dt = gw.sum(1)                                          # (T,) type mass
        sw = smem * Dt.unsqueeze(0)                             # (S, T)
        Smat = sw / sw.sum(1, keepdim=True).clamp_min(1e-6)
        Ds = sw.sum(1)                                          # (S,) super mass
        Rvec = Ds / Ds.sum().clamp_min(1e-6)                    # (S,)
        self.register_buffer("Gmat", Gmat)
        self.register_buffer("Smat", Smat)
        self.register_buffer("Rvec", Rvec)
        if finetune:
            self.register_buffer("Gmask", gmem)
            self.register_buffer("Smask", smem)
            # start fine-tuning from the informed (discriminativeness) prior:
            # masked-softmax(log w) reproduces w, then data adjusts it.
            self.Glogit = nn.Parameter(torch.log(Gmat.clamp_min(1e-6)))
            self.Slogit = nn.Parameter(torch.log(Smat.clamp_min(1e-6)))
            # PER-SPECIES super-category weight: each species learns its own
            # emphasis over super-categories (e.g. "this bird relies on colour
            # even more"), starting from the shared discriminativeness prior.
            self.Rlogit = nn.Parameter(
                torch.log(Rvec.clamp_min(1e-6)).unsqueeze(0).repeat(L, 1))

    @staticmethod
    def _mem(child_of_parent, n_parent):
        M = torch.zeros(n_parent, len(child_of_parent))
        for c, p in enumerate(child_of_parent):
            M[p, c] = 1.0
        return M

    @staticmethod
    def _masked_softmax(logit, mask):
        z = logit.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(z, dim=1)
        return torch.nan_to_num(w, nan=0.0)

    def forward(self, x, use_side=True, harden=False):
        c = torch.sigmoid(self.concept(self.backbone(x)))       # (B, K)
        cc_in = (c > 0.5).float() if harden else c
        P = self.sign.unsqueeze(0)                              # (1, L, K)
        cc = cc_in.unsqueeze(1)                                 # (B, 1, K)
        m = (P * cc + (1.0 - P) * (1.0 - cc)).clamp_min(1e-6)   # (B, L, K)
        logm = torch.log(m)
        if self.finetune:
            G = self._masked_softmax(self.Glogit, self.Gmask)   # (T, K)
            Smat = self._masked_softmax(self.Slogit, self.Smask)  # (S, T)
            R = torch.softmax(self.Rlogit, dim=1)               # (L, S) per-species
        else:
            G, Smat, R = self.Gmat, self.Smat, self.Rvec
        type_log = torch.einsum("blk,tk->blt", logm, G)         # (B, L, T)
        sc_log = torch.einsum("blt,st->bls", type_log, Smat)    # (B, L, S)
        if R.dim() == 2:
            root_log = torch.einsum("bls,ls->bl", sc_log, R)    # (B, L) per-species
        else:
            root_log = torch.einsum("bls,s->bl", sc_log, R)     # (B, L)
        truth = torch.exp(root_log).clamp(1e-6, 1.0 - 1e-6)
        logits = self.log_temp.exp() * (torch.log(truth) - torch.log1p(-truth))
        return logits, c


def write_cub_trees_json(path, spec=None):
    """Emit the 200 prompt-style hierarchical GL trees as JSON (for the record).

    Structure per species: SC over super-categories -> SC over attribute types ->
    signed concept leaves (positive where the prototype has the attribute ON, else
    negated).  Evaluation uses the vectorized _CUBGLOCBM (same math).
    """
    import json
    if spec is None:
        spec = build_spec()
    P = spec.Y                                                  # (200, 112) in {0,1}
    names, toc, tnames, sot = _load_attr_groups()
    concept_names = [f"c{j}" for j in range(P.shape[1])]
    # group concept indices by type, and type indices by super-category
    types = {t: [j for j in range(len(toc)) if toc[j] == t] for t in range(len(tnames))}
    supers = {s: [t for t in range(len(tnames)) if sot[t] == s] for s in range(len(_SUPERCATS))}
    # discriminativeness weight per concept = variance of its membership across
    # species (prompt principle 3); propagated up so groups are weighted by mass.
    import numpy as _np
    pbar = _np.asarray(P, dtype=float).mean(0)
    disc = _np.clip(pbar * (1.0 - pbar), 1e-4, None)
    trees = {}
    for k in range(P.shape[0]):
        sc_nodes = []
        s_mass = {s: float(sum(disc[j] for t in tlist for j in types[t]))
                  for s, tlist in supers.items()}
        s_tot = sum(s_mass.values()) or 1.0
        for s, tlist in supers.items():
            t_nodes = []
            t_mass = {t: float(sum(disc[j] for j in types[t])) for t in tlist}
            t_tot = sum(t_mass.values()) or 1.0
            for t in tlist:
                d_tot = float(sum(disc[j] for j in types[t])) or 1.0
                leaves = [{"concept": concept_names[j],
                           "weight": round(float(disc[j]) / d_tot, 4),
                           "negate": bool(P[k, j] < 0.5)} for j in types[t]]
                t_nodes.append({"op": "SC", "name": tnames[t],
                                "weight": round(t_mass[t] / t_tot, 4), "children": leaves})
            sc_nodes.append({"op": "SC", "name": _SUPERCATS[s],
                             "weight": round(s_mass[s] / s_tot, 4), "children": t_nodes})
        trees[str(k)] = {"op": "SC", "name": f"species_{k}", "children": sc_nodes}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trees, f)
    return path


def _build_model(key, spec):
    if key == "blackbox":
        return BlackBox(feat_dim=512, n_classes=200, backbone=_make_resnet()), False, False
    if key == "cbm":
        return SoftCBM(spec, feat_dim=512, backbone=_make_resnet()), True, False
    if key == "cbm+sc":
        return SoftCBMSC(spec, feat_dim=512, dropout_p=0.9, backbone=_make_resnet()), True, True
    if key in ("cream", "cream-wo-sc"):
        return CREAM(spec, feat_dim=512, dropout_p=0.9, backbone=_make_resnet()), True, True
    if key == "ocbm":
        return _CUBGLOCBM(spec.Y, _make_resnet(), finetune=False), True, False
    if key == "ocbm-ft":
        return _CUBGLOCBM(spec.Y, _make_resnet(), finetune=True), True, False
    raise ValueError(key)


def _train(model, loader, device, epochs, lam, has_c,
           backbone_lr=1e-4, head_lr=1e-3):
    model.to(device).train()
    bb_ids = {id(p) for p in model.backbone.parameters()}
    bb = [p for p in model.parameters() if id(p) in bb_ids]
    heads = [p for p in model.parameters() if id(p) not in bb_ids]
    opt = torch.optim.Adam([{"params": bb, "lr": backbone_lr},
                            {"params": heads, "lr": head_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for _ in range(epochs):
        for img, c, y in loader:
            img, c, y = img.to(device), c.to(device), y.to(device)
            logits, cprobs = model(img)
            loss = F.cross_entropy(logits, y)
            if has_c and cprobs is not None:
                loss = loss + lam * F.binary_cross_entropy(cprobs, c)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def _evaluate(model, loader, device, spec, has_side):
    model.eval()
    full_c = full_n = cpath_c = 0
    cacc_sum = cacc_n = 0
    for img, c, y in loader:
        img, c, y = img.to(device), c.to(device), y.to(device)
        logits, cprobs = model(img)
        full_c += (logits.argmax(1) == y).sum().item()
        full_n += y.numel()
        logits_cp = model(img, use_side=False)[0] if has_side else logits
        cpath_c += (logits_cp.argmax(1) == y).sum().item()
        if cprobs is not None:
            cacc_sum += concept_accuracy(cprobs, c, spec) * y.numel()
            cacc_n += y.numel()
    concept = (cacc_sum / cacc_n) if cacc_n else float("nan")
    return full_c / full_n, cpath_c / full_n, concept


def _run_ctruey(spec, device, iters, epochs=40, seed=0):
    """Linear probe on true concepts -> class (no backbone). ~100% ceiling."""
    from cub_trees import load_split
    Ctr, ytr = load_split("train")
    Cte, yte = load_split("test")
    tl = DataLoader(TensorDataset(Ctr, ytr), batch_size=128, shuffle=True)
    accs = []
    for it in range(iters):
        _bench.set_seed(seed + it)
        model = CtrueY(spec).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(epochs):
            for c, y in tl:
                c, y = c.to(device), y.to(device)
                loss = F.cross_entropy(model(c)[0], y)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Cte.to(device))[0].argmax(1).cpu()
        accs.append((pred == yte).float().mean().item())
        print(f"  [iter {it + 1}/{iters}]  ACC_Y = {accs[-1] * 100:.2f}")
    return accs


def run_cell(model_key: str, iters: int, epochs: int = 30,
             batch_size: int = 64, lam: float = 1.0, seed: int = 0,
             num_workers: int = 4, ay_mode: str = "positive"):
    device = _bench.get_device()
    spec = build_spec(ay_mode)
    label = LABELS.get(model_key, model_key)
    print(f"{label} / CUB   iters={iters}  epochs={epochs}  K={spec.K}  "
          f"lam={lam}  ay={ay_mode}  device={device}")

    if model_key == "ctruey":
        return _run_ctruey(spec, device, iters, seed=seed), None

    tl = DataLoader(_CUBImages("train", True), batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    vl = DataLoader(_CUBImages("test", False), batch_size=128, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    acc_y, acc_c = [], []
    for it in range(iters):
        _bench.set_seed(seed + it)
        model, has_c, has_side = _build_model(model_key, spec)
        _train(model, tl, device, epochs, lam, has_c)
        full, cpath, cacc = _evaluate(model, vl, device, spec, has_side)
        ay = cpath if model_key == "cream-wo-sc" else full
        acc_y.append(ay)
        if has_c:
            acc_c.append(cacc)
        cstr = f"  ACC_C = {cacc * 100:.2f}" if has_c else ""
        print(f"  [iter {it + 1}/{iters}]  ACC_Y = {ay * 100:.2f}{cstr}")

    return acc_y, (acc_c or None)
