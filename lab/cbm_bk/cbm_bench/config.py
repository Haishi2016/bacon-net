"""Experiment configuration: load YAML into a typed structure.

Config files live under ``lab/cbm/configs``. Each file fully describes one
(dataset, model, training) experiment so runs are reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class ExperimentConfig:
    experiment: str
    seed: int
    device: str
    dataset: Dict[str, Any]
    model: Dict[str, Any]
    train: Dict[str, Any]
    output: Dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> str:
        return self.output.get("dir", "runs")


def load_config(path: str) -> ExperimentConfig:
    """Parse a YAML experiment config from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = ExperimentConfig(
        experiment=raw.get("experiment", os.path.splitext(os.path.basename(path))[0]),
        seed=int(raw.get("seed", 42)),
        device=raw.get("device", "auto"),
        dataset=raw["dataset"],
        model=raw["model"],
        train=raw["train"],
        output=raw.get("output", {}),
    )

    # Resolve a relative dataset root against the config file's directory so
    # configs are portable regardless of the current working directory.
    root = cfg.dataset.get("root")
    if root and not os.path.isabs(root):
        base = os.path.dirname(os.path.abspath(path))
        cfg.dataset["root"] = os.path.normpath(os.path.join(base, root))
    return cfg
