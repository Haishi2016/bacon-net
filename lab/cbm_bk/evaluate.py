#!/usr/bin/env python3
"""Evaluate a trained CBM checkpoint on the test split.

Usage
-----
    python lab/cbm/evaluate.py \
        --config lab/cbm/configs/cub_baseline.yaml \
        --checkpoint runs/cub_baseline/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cbm_bench import build_dataset, build_model  # noqa: E402
from cbm_bench.config import load_config  # noqa: E402
from cbm_bench.concept_purity import (  # noqa: E402
    niche_impurity_score,
    oracle_impurity_score,
)
from cbm_bench.engine import (  # noqa: E402
    collect_concepts,
    make_loaders,
    resolve_device,
    run_epoch,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CBM checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument(
        "--no-purity",
        action="store_true",
        help="Skip the OIS / NIS concept-purity metrics (they train MLP probes "
        "and can be slow for many concepts).",
    )
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Measure concept purity on the SAMPLED concept channel rather than "
        "the deterministic mean (variational CIBM only). The backbone still "
        "runs deterministically; only q(c|x) is sampled. Use this to compare "
        "noisy-channel leakage against the mean-channel leakage.",
    )
    parser.add_argument(
        "--purity-instance",
        action="store_true",
        help="Compute OIS / NIS against INSTANCE-level (raw per-image) concept "
        "labels instead of the class-majority prototypes used for training. "
        "Under class_majority every image of a class shares one identical "
        "concept vector, so the ground-truth concepts are a deterministic "
        "function of the class -- this collapses the cross-concept correlation "
        "structure the leakage metrics rely on and inflates OIS/NIS regardless "
        "of representation quality. The purity metrics are defined on per-image "
        "annotations (Zarlenga et al. 2023), so use this flag for comparable "
        "numbers. Only the purity probes' ground-truth changes; task / concept "
        "accuracy and the model's soft predictions are unaffected.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    bundle = build_dataset(cfg.dataset["name"], cfg.dataset)
    model = build_model(
        cfg.model["name"], cfg.model, bundle.n_concepts, bundle.n_classes
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # ``strict=False`` tolerates checkpoints saved before the routing-state
    # buffers (temperature / gumbel_noise_scale) were added; any genuinely
    # missing weight would still surface as a poor result.
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
    missing = [k for k in incompatible.missing_keys]
    unexpected = [k for k in incompatible.unexpected_keys]
    if missing:
        print(f"[warn] missing keys not in checkpoint: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys in checkpoint: {unexpected}")
    # A trained model must be evaluated at its committed (fully annealed) routing.
    # The Sinkhorn temperature is broad early in training; evaluating at the
    # initial temperature gives near-uniform routing and collapses accuracy.
    if hasattr(model, "anneal"):
        model.anneal(1.0)
    print(f"[loaded] {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    loaders = make_loaders(bundle, cfg.train)
    concept_w = float(cfg.model.get("concept_loss_weight", 1.0))
    concept_reduction = str(cfg.model.get("concept_reduction", "mean")).lower()
    stats = run_epoch(
        model, loaders["test"], device, concept_w,
        concept_reduction=concept_reduction,
    )

    print("[test results]")
    print(f"  loss        : {stats['loss']:.4f}")
    print(f"  task acc    : {stats['task_acc']:.4f}")
    print(f"  concept acc : {stats['concept_acc']:.4f}")

    if not args.no_purity:
        # Concept-leakage metrics (Zarlenga et al. 2023). Lower == purer.
        # Computed on the test split's soft concept probabilities vs the
        # ground-truth concept labels.
        channel = "sampled" if args.stochastic_eval else "deterministic mean"
        # The leakage metrics need per-image ground-truth concepts (their
        # cross-concept correlation structure). Under class_majority every image
        # of a class shares one concept vector, which degenerates that structure
        # and inflates OIS/NIS. When requested, rebuild the test set with
        # instance-level concepts purely for the probes; the model and its soft
        # predictions are unchanged (only the loader's c_true differs).
        purity_loader = loaders["test"]
        if args.purity_instance:
            if str(cfg.dataset.get("concept_mode", "class_majority")) == "instance":
                print("[purity] config already uses instance concepts; "
                      "--purity-instance is a no-op")
            else:
                inst_cfg = dict(cfg.dataset)
                inst_cfg["concept_mode"] = "instance"
                inst_bundle = build_dataset(cfg.dataset["name"], inst_cfg)
                purity_loader = make_loaders(inst_bundle, cfg.train)["test"]
                print("[purity] using INSTANCE-level concept labels for OIS / NIS")
        c_soft, c_true = collect_concepts(
            model, purity_loader, device, stochastic=args.stochastic_eval
        )
        print(
            f"[concept purity] computing OIS / NIS over "
            f"{c_soft.shape[0]} samples x {c_soft.shape[1]} concepts "
            f"on the {channel} channel "
            f"(this trains MLP probes; use --no-purity to skip)"
        )
        ois = oracle_impurity_score(c_soft, c_true)
        nis = niche_impurity_score(c_soft, c_true)
        print(f"  OIS         : {ois:.4f}  (oracle impurity, lower is purer)")
        print(f"  NIS         : {nis:.4f}  (niche impurity, lower is purer)")


if __name__ == "__main__":
    main()
