#!/usr/bin/env python3
"""Train a CBM model on a dataset described by a YAML config.

Usage
-----
    python lab/cbm/train.py --config lab/cbm/configs/cub_baseline.yaml

Checkpoints and a metrics log are written to ``<output.dir>/<experiment>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# Make ``cbm_bench`` importable when run as a script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cbm_bench import build_dataset, build_model  # noqa: E402
from cbm_bench.config import load_config  # noqa: E402
from cbm_bench.engine import (  # noqa: E402
    make_loaders,
    resolve_device,
    run_epoch,
    set_seed,
)


def build_optimizer(model, train_cfg) -> torch.optim.Optimizer:
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    name = train_cfg.get("optimizer", "adam").lower()
    # Pretrained backbone wants a small LR; freshly initialized heads (concept
    # head + BACON logic head) need a larger one. ``head_lr_mult`` scales the
    # head LR relative to the backbone; models exposing ``param_groups`` opt in.
    head_lr_mult = float(train_cfg.get("head_lr_mult", 1.0))
    if head_lr_mult != 1.0 and hasattr(model, "param_groups"):
        params = model.param_groups(lr, head_lr_mult)
    else:
        params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'.")


def build_scheduler(optimizer, train_cfg):
    name = train_cfg.get("scheduler")
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("step_size", 30)),
            gamma=float(train_cfg.get("gamma", 0.1)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(train_cfg.get("epochs", 100))
        )
    return None


def _class_concept_prototypes(dataset, n_classes: int, n_concepts: int):
    """Mean concept vector per class from a dataset's ``(path, concepts, label)``.

    Returns ``(n_classes, n_concepts)`` in ``[0, 1]`` (the per-class concept
    signature), or ``None`` if the dataset doesn't expose precomputed samples.
    Reads only the cached concept tensors, so no images are decoded.
    """
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None
    sums = torch.zeros(n_classes, n_concepts)
    counts = torch.zeros(n_classes, 1)
    for _, concept_vec, label in samples:
        sums[label] += concept_vec
        counts[label, 0] += 1
    # Classes with no samples stay at 0.5 (neutral gate).
    proto = torch.full((n_classes, n_concepts), 0.5)
    seen = counts.squeeze(1) > 0
    proto[seen] = sums[seen] / counts[seen]
    return proto


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CBM benchmark model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    out_dir = os.path.join(cfg.output_dir, cfg.experiment)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[config]  {args.config}")
    print(f"[device]  {device}")
    print(f"[dataset] {cfg.dataset['name']}  root={cfg.dataset.get('root')}")

    bundle = build_dataset(cfg.dataset["name"], cfg.dataset)
    model = build_model(
        cfg.model["name"], cfg.model, bundle.n_concepts, bundle.n_classes
    ).to(device)

    loaders = make_loaders(bundle, cfg.train)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train)
    concept_w = float(cfg.model.get("concept_loss_weight", 1.0))
    # How the per-concept BCE terms are reduced. "sum" (sum over concepts, mean
    # over batch) is the faithful CBM objective where concept_loss_weight: 1.0
    # truly weights every CE term equally (Koh et al. eq. 6 / CIBM paper);
    # "mean" (default) keeps the legacy scale used by the tuned harness configs.
    concept_reduction = str(cfg.model.get("concept_reduction", "mean")).lower()
    # Label smoothing on the task cross-entropy (0 disables). Regularises the
    # task head against overfitting the ~24-img/class CUB train set without
    # slowing concept learning; only affects the default CE (BACON's own
    # task_loss ignores it). Default 0.0 keeps the tuned harness configs intact.
    label_smoothing = float(cfg.train.get("label_smoothing", 0.0))
    grad_clip = cfg.train.get("grad_clip", 1.0)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    # Optionally warm-start the vector head's per-class relevance gates from
    # each class's concept signature. This is OFF by default: the task-loss
    # reduction (mean over batch, sum over heads) already lets cold-started
    # heads learn. When enabled (model.warm_start_gates: true) it seeds each
    # head at its class's interpretable concept signature, which can speed
    # convergence and improve rule interpretability. No-op for other models.
    if cfg.model.get("warm_start_gates", False) and hasattr(
        model, "init_head_gates_from_prototypes"
    ):
        proto = _class_concept_prototypes(
            bundle.train, bundle.n_classes, bundle.n_concepts
        )
        if proto is not None and model.init_head_gates_from_prototypes(proto):
            print("[init]    warm-started vector head gates from class prototypes")

    eval_split = "val" if "val" in loaders else "test"
    epochs = int(cfg.train.get("epochs", 100))
    # Number of epochs over which to anneal the BACON head's routing schedule
    # (broad -> sharp). Defaults to the full run, but finishing the anneal early
    # (anneal_epochs < epochs) leaves an exploitation phase where the head trains
    # under its committed near-permutation instead of only reaching it at the
    # final step. ``progress`` saturates at 1.0 once ``anneal_epochs`` is reached.
    anneal_epochs = int(cfg.train.get("anneal_epochs", epochs))
    history = []
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Advance the BACON head's exploration schedule (broad -> sharp routing).
        # progress 0.0 -> 1.0 over the anneal window; no-op for models without ``anneal``.
        if hasattr(model, "anneal"):
            progress = min((epoch - 1) / max(anneal_epochs - 1, 1), 1.0)
            model.anneal(progress)
        train_stats = run_epoch(
            model, loaders["train"], device, concept_w, optimizer, grad_clip,
            concept_reduction, label_smoothing,
        )
        eval_stats = run_epoch(
            model, loaders[eval_split], device, concept_w,
            concept_reduction=concept_reduction,
        )
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_stats,
            eval_split: eval_stats,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(time.time() - t0, 1),
        }
        history.append(record)
        print(
            f"[{epoch:3d}/{epochs}] "
            f"train loss {train_stats['loss']:.4f} "
            f"task {train_stats['task_acc']:.3f} | "
            f"{eval_split} task {eval_stats['task_acc']:.3f} "
            f"concept {eval_stats['concept_acc']:.3f} "
            f"({record['time_s']}s)"
        )

        if eval_stats["task_acc"] > best_acc:
            best_acc = eval_stats["task_acc"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "config": args.config},
                os.path.join(out_dir, "best.pt"),
            )

    torch.save(
        {"epoch": epochs, "model_state": model.state_dict(), "config": args.config},
        os.path.join(out_dir, "last.pt"),
    )

    # Final held-out test evaluation of the *selected* checkpoint. When a
    # validation split exists, model selection happens on val (paper protocol),
    # so test was never seen during training -- evaluate the best-on-val weights
    # on test here to report the unbiased number the paper quotes. When there is
    # no val split, eval_split is already "test" and best_acc is that number.
    test_stats = None
    if eval_split != "test" and "test" in loaders:
        best_path = os.path.join(out_dir, "best.pt")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        test_stats = run_epoch(
            model, loaders["test"], device, concept_w,
            concept_reduction=concept_reduction,
        )
        history.append({"final_test": test_stats})

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(f"[done] best {eval_split} task acc {best_acc:.4f} -> {out_dir}")
    if test_stats is not None:
        print(
            f"[test] selected-checkpoint test task {test_stats['task_acc']:.4f} "
            f"concept {test_stats['concept_acc']:.4f}"
        )

    # Persist the 200 per-class BACON logic trees next to the checkpoint so the
    # standard single-tree visualization / analysis tooling can read them. The
    # vectorized head packs every class rule into batched tensors; this unpacks
    # the *selected* (best) checkpoint into one scalar-tree JSON per class under
    # ``<out_dir>/trees/``. No-op for non-BACON models.
    if hasattr(model, "extract_all_scalar_trees") and getattr(model, "vector_net", None) is not None:
        best_path = os.path.join(out_dir, "best.pt")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        from cbm_bench.models.bacon_cbm import save_bacon_trees

        trees_dir = os.path.join(out_dir, "trees")
        paths = save_bacon_trees(
            model.to("cpu"),
            trees_dir,
            concept_names=getattr(bundle, "concept_names", None),
            class_names=getattr(bundle, "class_names", None),
        )
        print(f"[trees] saved {len(paths)} per-class logic trees -> {trees_dir}")



if __name__ == "__main__":
    main()
