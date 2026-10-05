"""Train a BACON model from the portal and emit progress + the learned tree.

This mirrors samples/hello-world/main.py: it uses the real `bacon` library
(baconNet + find_best_model) with the same configuration, trains on a dataset
CSV, then exports the learned tree structure via
`bacon.utils.export_tree_structure_to_json` and converts it to the nested
{label,count,children} shape the portal tree editor consumes.

Usage:
    python train_bacon.py --csv <path-to-csv> [--max-epochs N] [--attempts K]

Output protocol (one item per line on stdout):
    LOG::<message>            human-readable progress line (streamed to the UI)
    TREE::<json>             the learned tree (editor format) — emitted once
    DONE::<json>             final summary {accuracy,...}
    ERROR::<message>         a fatal error
"""

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

# Make the local `bacon` package importable (repo root is two levels up from
# portal/scripts/). Allow an explicit override via --repo for safety.
DEFAULT_REPO = Path(__file__).resolve().parents[2]

LABEL_NAMES = {"label", "target", "y", "class", "output"}


def emit(channel: str, payload: str) -> None:
    """Write a single protocol line and flush so the UI streams it live."""
    sys.stdout.write(f"{channel}::{payload}\n")
    sys.stdout.flush()


def _force_utf8() -> None:
    # bacon's logs (and ours) contain emoji; Windows consoles default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


class EmitLogHandler(logging.Handler):
    """Forward bacon's INFO logging (epoch loss, attempts, accuracy) to the UI."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            emit("LOG", self.format(record))
        except Exception:
            pass


def load_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError("CSV has no data rows.")

    header = [cell.strip() for cell in rows[0]]
    # Treat a trailing label-like column as the target; otherwise the last column.
    label_idx = len(header) - 1
    feature_names = header[:label_idx]

    features = []
    labels = []
    for row in rows[1:]:
        if len(row) <= label_idx:
            continue
        features.append([float(row[i]) for i in range(label_idx)])
        labels.append([float(row[label_idx])])
    return feature_names, features, labels


def to_editor_tree(structure: dict, graded: bool = False):
    """Convert export_tree_structure_to_json output to the editor's nested form.

    When `graded` is True, aggregator nodes are labeled with the named GCD
    operator code (e.g. SC+, CP) matching the palette; otherwise they use the
    discrete AND/OR label (for bool / operator-set families).
    """

    def op_label(andness: float) -> str:
        if graded:
            from bacon.utils import andness_to_gcd_code
            return andness_to_gcd_code(andness)
        return "AND" if andness >= 0.5 else "OR"

    def feature_node(name: str) -> dict:
        return {"label": name, "count": 1}

    layout = structure.get("layout")

    if layout == "left":
        nodes = structure.get("nodes", [])
        if not nodes:
            return [feature_node(f["display_name"]) for f in structure.get("features", [])]

        def build(index: int) -> dict:
            node = nodes[index]
            andness = node["andness"]
            op = op_label(andness)
            left_in = node["left_input"]
            right_in = node["right_input"]
            left = build(left_in["layer"]) if left_in["type"] == "aggregator" else feature_node(left_in["name"])
            right = feature_node(right_in["name"]) if right_in["type"] == "feature" else build(right_in["layer"])
            return {
                "label": op,
                "count": left["count"] + right["count"],
                "operator": op,
                "andness": round(andness, 3),
                "children": [left, right],
            }

        return [build(len(nodes) - 1)]

    if "root" in structure:

        def build(node: dict) -> dict:
            if node.get("type") == "feature":
                return feature_node(node["name"])
            andness = node["andness"]
            op = op_label(andness)
            left = build(node["left_input"])
            right = build(node["right_input"])
            return {
                "label": op,
                "count": left["count"] + right["count"],
                "operator": op,
                "andness": round(andness, 3),
                "children": [left, right],
            }

        return [build(structure["root"])]

    # Fallback: flat list of features.
    return [feature_node(f["display_name"]) for f in structure.get("features", [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--aggregator", default="bool.min_max")
    parser.add_argument("--head-type", default="left", choices=["left", "rect"])
    parser.add_argument("--rect-depth", type=int, default=4)
    parser.add_argument("--save-model", default="", help="Path to save the trained model (.pth) via bacon's save_model")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=0, help="0 = auto (min(input*300, 8000))")
    args = parser.parse_args()

    _force_utf8()
    sys.path.insert(0, args.repo)

    # Stream bacon's own INFO logs (epochs, attempts, accuracy) to the UI.
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[EmitLogHandler()])

    try:
        import torch
        from bacon.baconNet import baconNet
        from bacon.utils import export_tree_structure_to_json
    except Exception as exc:  # noqa: BLE001
        emit("ERROR", f"Could not import bacon/torch: {exc}")
        return 1

    try:
        feature_names, features, labels = load_csv(Path(args.csv))
    except Exception as exc:  # noqa: BLE001
        emit("ERROR", f"Could not read CSV: {exc}")
        return 1

    input_size = len(feature_names)
    if input_size < 2:
        emit("ERROR", "Need at least 2 feature columns to train.")
        return 1

    emit("LOG", f"📦 Loaded {len(features)} samples × {input_size} features: {', '.join(feature_names)}")

    seed = 7
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)

    def count_nodes(nodes):
        total = 0
        for node in nodes:
            total += 1 + count_nodes(node.get("children", []))
        return total

    if args.head_type == "rect":
        # Rectangular graded-logic head pruned into a tree during training.
        from bacon.vectorizedRectTree import VectorRectTreeHead

        depth = max(1, int(args.rect_depth))
        y_vec = y.squeeze(-1) if y.dim() > 1 else y
        epochs = args.max_epochs if args.max_epochs > 0 else 600
        emit("LOG", f"🏋️ Training rect head · depth={depth} · width={input_size} · epochs={epochs}")

        # The real OCBM single-run harden protocol (anneal -> L0 edge warmup ->
        # binarize warmup -> harden@freeze_frac -> frozen fine-tune) lives in the
        # bacon library (VectorRectTreeHead.train / .fit), the same machinery
        # behind run_cub / run_awa / run_cifar. Here we just drive it and relay
        # progress to the UI. No multiple attempts.
        def on_progress(info):
            phase = info.get("phase")
            if phase == "soft":
                emit(
                    "LOG",
                    f"   🏋️ Epoch {info['epoch']}/{info['epochs']} - Loss: {info['loss']:.4f} - Acc: {info['acc'] * 100:.1f}%",
                )
            elif phase == "harden":
                emit("LOG", f"✂️ Hardened into a tree · accuracy: {info['acc'] * 100:.2f}%")
            elif phase == "finetune":
                emit("LOG", f"   🔧 Fine-tune {info['epoch']}/{info['epochs']} - Acc: {info['acc'] * 100:.1f}%")

        try:
            head, result = VectorRectTreeHead.train_head(
                x,
                y_vec,
                input_size=input_size,
                depth=depth,
                max_parents=1,
                use_negation=True,
                normalize_andness=True,
                temperature=2.0,
                final_temperature=0.1,
                epochs=epochs,
                progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001
            emit("ERROR", f"Training failed: {exc}")
            return 1

        soft_accuracy = result["soft_accuracy"]
        hard_accuracy = result["hard_accuracy"]
        final_accuracy = result["final_accuracy"]
        emit("LOG", f"📏 soft {soft_accuracy * 100:.2f}% · hardened {hard_accuracy * 100:.2f}% · final {final_accuracy * 100:.2f}%")

        try:
            editor_tree = head.export_hardened_tree(feature_names)
        except Exception as exc:  # noqa: BLE001
            emit("ERROR", f"Could not export learned tree: {exc}")
            return 1

        node_count = count_nodes(editor_tree)
        link_count = max(0, node_count - len(editor_tree))
        metadata = {
            "feature_names": feature_names,
            "head_type": "rect",
            "rect_depth": depth,
            "aggregator": "lsp.full_weight",
            "accuracy": round(final_accuracy, 4),
            "hard_accuracy": round(hard_accuracy, 4),
            "tree": editor_tree,
            "nodes": node_count,
            "links": link_count,
        }
        if args.save_model:
            try:
                head.save_model(args.save_model, metadata=metadata)
                emit("LOG", f"💾 Saved model checkpoint: {Path(args.save_model).name}")
                emit("MODEL", Path(args.save_model).name)
            except Exception as exc:  # noqa: BLE001
                emit("LOG", f"⚠️ Could not save model checkpoint: {exc}")

        emit("TREE", json.dumps(editor_tree, ensure_ascii=False))
        emit("DONE", json.dumps({"accuracy": round(final_accuracy, 4), "hardAccuracy": round(hard_accuracy, 4)}))
        return 0

    # Left-associative tree (default), as in samples/hello-world/main.py.
    bacon = baconNet(
        input_size,
        aggregator=args.aggregator,
        tree_layout="left",
        weight_mode="fixed",
        loss_amplifier=1000,
        normalize_andness=False,
        use_permutation_layer=False,
    )

    max_epochs = args.max_epochs if args.max_epochs > 0 else min(input_size * 300, 8000)
    emit("LOG", f"🏋️ Training BACON · aggregator={args.aggregator} · attempts={args.attempts} · max_epochs={max_epochs}")

    try:
        best_model, best_accuracy = bacon.find_best_model(
            x, y, x, y,
            acceptance_threshold=0.95,
            attempts=args.attempts,
            max_epochs=max_epochs,
            save_model=False,
        )
    except Exception as exc:  # noqa: BLE001
        emit("ERROR", f"Training failed: {exc}")
        return 1

    emit("LOG", f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")

    pred = bacon.inference(x, threshold=0.5)
    final_accuracy = (pred == y).float().mean().item()
    emit("LOG", f"📏 Final accuracy: {final_accuracy * 100:.2f}%")

    try:
        structure = export_tree_structure_to_json(bacon.assembler, feature_names)
        graded = args.aggregator.startswith("lsp.") or args.aggregator == "gl.generic"
        editor_tree = to_editor_tree(structure, graded=graded)
    except Exception as exc:  # noqa: BLE001
        emit("ERROR", f"Could not export learned tree: {exc}")
        return 1

    node_count = count_nodes(editor_tree)
    link_count = max(0, node_count - len(editor_tree))

    # Persist the trained model using bacon's own save feature (a .pth checkpoint).
    # Display metadata (feature names, aggregator, accuracy, tree) is embedded in
    # the checkpoint so a single .pth supports display/train/inference.
    if args.save_model:
        metadata = {
            "feature_names": feature_names,
            "head_type": "left",
            "aggregator": args.aggregator,
            "accuracy": round(final_accuracy, 4),
            "best_accuracy": round(best_accuracy, 4),
            "tree": editor_tree,
            "nodes": node_count,
            "links": link_count,
        }
        try:
            bacon.save_model(args.save_model, metadata=metadata)
            emit("LOG", f"💾 Saved model checkpoint: {Path(args.save_model).name}")
            emit("MODEL", Path(args.save_model).name)
        except Exception as exc:  # noqa: BLE001
            emit("LOG", f"⚠️ Could not save model checkpoint: {exc}")

    emit("TREE", json.dumps(editor_tree, ensure_ascii=False))
    emit("DONE", json.dumps({"accuracy": round(final_accuracy, 4), "bestAccuracy": round(best_accuracy, 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
