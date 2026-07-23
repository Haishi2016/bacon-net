"""
FashionMNIST concept definitions for the CREAM reproduction + our BACON trees.

Three settings:

  iFMNIST (incomplete, K=8) : hierarchical apparel categories only.  {T-shirt,
    Pullover, Shirt} and {Sandal, Sneaker, Ankle boot} share identical concept
    vectors -> concepts cap task accuracy at ~60% (paper Table 5).

  cFMNIST (complete, K=11)  : CREAM's fix -- adds a mutually-exclusive *season*
    group {Summer, Winter, Mild} that one-hot-disambiguates the ambiguous
    triples -> concepts fully determine the class (~100%).

  sFMNIST (complete, K=12)  : OUR fix -- instead of an artificial season group,
    add four *semantically meaningful binary* attributes
    {long_sleeve, front_opening, open_toe, ankle_high} and disambiguate with
    SOPHISTICATED BACON trees that use NOT / OR / nesting, e.g.
        Pullover = Clothes AND Tops AND long_sleeve AND NOT front_opening
        Sneaker  = Goods   AND Shoes AND NOT (open_toe OR ankle_high)
    This shows incompleteness can be resolved with richer *logic* over a few
    interpretable concepts, not just more one-hot concepts.

Concept types:
  * mutex_groups  : mutually-exclusive concept groups (softmax activation).
  * binary_concepts: independent binary attributes (sigmoid activation).

Each ConceptSpec is JSON-serialisable (to_dict / from_dict) so the trees can be
saved and reused.
"""

from __future__ import annotations

import torch

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]


class ConceptSpec:
    def __init__(self, name, concept_names, mutex_groups, binary_concepts,
                 class_on, formulas=None):
        self.name = name
        self.concept_names = list(concept_names)
        self.mutex_groups = [list(g) for g in mutex_groups]
        self.binary_concepts = list(binary_concepts)
        self.groups = self.mutex_groups                    # backward-compat alias
        self.index = {n: i for i, n in enumerate(self.concept_names)}
        self.K = len(self.concept_names)
        self.class_on = {int(c): list(v) for c, v in class_on.items()}

        # class -> binary concept ground-truth vector (10, K)
        self.Y = torch.zeros(10, self.K)
        for c, on in self.class_on.items():
            for p in on:
                self.Y[c, self.index[p]] = 1.0
        self.A_Y = self.Y.clone()                          # (L=10, K): class -> concept

        # per-class BACON tree (defaults to AND of the ON concepts)
        if formulas is None:
            formulas = {c: " AND ".join(on) for c, on in self.class_on.items()}
        self.formulas = {int(c): f for c, f in formulas.items()}

    def concept_targets(self, labels: torch.Tensor) -> torch.Tensor:
        return self.Y.to(labels.device)[labels]

    # -- serialisation -------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "concept_names": self.concept_names,
            "mutex_groups": self.mutex_groups,
            "binary_concepts": self.binary_concepts,
            "class_on": {str(c): v for c, v in self.class_on.items()},
            "formulas": {str(c): f for c, f in self.formulas.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptSpec":
        return cls(d["name"], d["concept_names"], d["mutex_groups"],
                   d["binary_concepts"], d["class_on"], d["formulas"])


# --------------------------------------------------------------------------- #
# iFMNIST (incomplete)
# --------------------------------------------------------------------------- #
_IF_CONCEPTS = ["Clothes", "Goods",
                "Tops", "Bottoms", "Dresses", "Outers", "Accessories", "Shoes"]
_IF_GROUPS = [[0, 1], [2, 3, 4, 5, 6, 7]]
_IF_ON = {
    0: ["Clothes", "Tops"], 1: ["Clothes", "Bottoms"], 2: ["Clothes", "Tops"],
    3: ["Clothes", "Dresses"], 4: ["Clothes", "Outers"], 5: ["Goods", "Shoes"],
    6: ["Clothes", "Tops"], 7: ["Goods", "Shoes"], 8: ["Goods", "Accessories"],
    9: ["Goods", "Shoes"],
}

# --------------------------------------------------------------------------- #
# cFMNIST (complete via season mutex group)
# --------------------------------------------------------------------------- #
_CF_CONCEPTS = _IF_CONCEPTS + ["Summer", "Winter", "Mild"]
_CF_GROUPS = [[0, 1], [2, 3, 4, 5, 6, 7], [8, 9, 10]]
_CF_SEASON = {0: "Summer", 1: "Mild", 2: "Winter", 3: "Summer", 4: "Winter",
              5: "Summer", 6: "Mild", 7: "Mild", 8: "Mild", 9: "Winter"}
_CF_ON = {c: _IF_ON[c] + [_CF_SEASON[c]] for c in range(10)}

# --------------------------------------------------------------------------- #
# sFMNIST (complete via meaningful binary attributes + sophisticated trees)
# --------------------------------------------------------------------------- #
_SF_CONCEPTS = _IF_CONCEPTS + ["long_sleeve", "front_opening", "open_toe", "ankle_high"]
_SF_GROUPS = [[0, 1], [2, 3, 4, 5, 6, 7]]           # apparel mutex groups
_SF_BINARY = [8, 9, 10, 11]                          # independent binary attrs
_SF_ON = {
    0: ["Clothes", "Tops"],                                       # T-shirt
    1: ["Clothes", "Bottoms"],                                    # Trouser
    2: ["Clothes", "Tops", "long_sleeve"],                        # Pullover
    3: ["Clothes", "Dresses"],                                    # Dress
    4: ["Clothes", "Outers", "long_sleeve", "front_opening"],     # Coat
    5: ["Goods", "Shoes", "open_toe"],                            # Sandal
    6: ["Clothes", "Tops", "long_sleeve", "front_opening"],       # Shirt
    7: ["Goods", "Shoes"],                                        # Sneaker
    8: ["Goods", "Accessories"],                                  # Bag
    9: ["Goods", "Shoes", "ankle_high"],                          # Ankle boot
}
_SF_FORMULAS = {
    0: "Clothes AND Tops AND NOT long_sleeve",
    1: "Clothes AND Bottoms",
    2: "Clothes AND Tops AND long_sleeve AND NOT front_opening",
    3: "Clothes AND Dresses",
    4: "Clothes AND Outers",
    5: "Goods AND Shoes AND open_toe",
    6: "Clothes AND Tops AND long_sleeve AND front_opening",
    7: "Goods AND Shoes AND NOT (open_toe OR ankle_high)",
    8: "Goods AND Accessories",
    9: "Goods AND Shoes AND ankle_high",
}


IFMNIST = ConceptSpec("iFMNIST", _IF_CONCEPTS, _IF_GROUPS, [], _IF_ON)
CFMNIST = ConceptSpec("cFMNIST", _CF_CONCEPTS, _CF_GROUPS, [], _CF_ON)
SFMNIST = ConceptSpec("sFMNIST", _SF_CONCEPTS, _SF_GROUPS, _SF_BINARY, _SF_ON, _SF_FORMULAS)

SPECS = {"iFMNIST": IFMNIST, "cFMNIST": CFMNIST, "sFMNIST": SFMNIST}
