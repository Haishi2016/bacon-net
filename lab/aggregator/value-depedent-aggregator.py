import argparse
import pathlib
import random
import sys
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# Ensure local repo package imports work when running this script directly.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bacon.aggregators.lsp.generic_gl import GenericGLAggregator

SEED = 42
EPS = 1e-8

# Anchor preset configurations
ANCHOR_PRESETS = {
    "simple": ("min", "mean", "max"),
    "extended": ("min", "harmonic", "mean", "max"),
    "full": ("min", "harmonic", "geometric", "mean", "quadratic", "max"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _apply_anchor_label_map(a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Convert region labels to scalar targets using only {min, mean, max} anchors.
    label 0 -> min(a,b), 1 -> mean(a,b), 2 -> max(a,b)
    """
    min_v = torch.minimum(a, b)
    mean_v = 0.5 * (a + b)
    max_v = torch.maximum(a, b)

    y = torch.where(labels == 0, min_v, max_v)
    y = torch.where(labels == 1, mean_v, y)
    return y


def split_labels_balanced_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Original split: x+y>1 -> min, x+y<1 -> max."""
    labels = torch.full_like(a, 2, dtype=torch.long)  # default=max
    labels = torch.where(a + b > 1.0, torch.zeros_like(labels), labels)  # min
    return labels


def split_labels_quarter_ring(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quarter-ring mean region: annulus 0.30<=r<=0.65 centred at origin is mean;
    interior (r<0.30) is max; exterior (r>0.65) is min.
    Off-diagonal by construction, so mean(a,b) truly differs from min and max.
    """
    r = torch.sqrt(a ** 2 + b ** 2)
    labels = torch.full_like(a, 0, dtype=torch.long)  # default=min (outer)
    labels = torch.where(r < 0.30, torch.full_like(labels, 2), labels)  # max (inner)
    labels = torch.where((r >= 0.30) & (r <= 0.65), torch.ones_like(labels), labels)  # mean (ring)
    return labels


def split_labels_islands(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex split with center-island mean region and wavy min pockets."""
    r2 = (a - 0.5) ** 2 + (b - 0.5) ** 2
    center_island = r2 < 0.06

    wave = torch.sin(2.6 * np.pi * a) * torch.cos(2.2 * np.pi * b)
    min_zone = (a + b > 1.0 + 0.10 * wave) & (~center_island)

    labels = torch.full_like(a, 2, dtype=torch.long)  # default=max
    labels = torch.where(min_zone, torch.zeros_like(labels), labels)  # min
    labels = torch.where(center_island, torch.ones_like(labels), labels)  # mean
    return labels


def scenario_balanced_sum_split(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    labels = split_labels_balanced_sum(a, b)
    return _apply_anchor_label_map(a, b, labels)


def scenario_quarter_ring(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    labels = split_labels_quarter_ring(a, b)
    return _apply_anchor_label_map(a, b, labels)


def scenario_island_wave_split(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    labels = split_labels_islands(a, b)
    return _apply_anchor_label_map(a, b, labels)


SCENARIOS: dict[str, tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = {
    "balanced": (
        "Balanced split: min if a+b>1 else max",
        scenario_balanced_sum_split,
    ),
    "curved": (
        "Quarter-ring: mean annulus (0.30<=r<=0.65), max inside, min outside",
        scenario_quarter_ring,
    ),
    "wave": (
        "Island split: center mean island + wavy min pockets",
        scenario_island_wave_split,
    ),
}

SCENARIO_LABELS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "balanced": split_labels_balanced_sum,
    "curved": split_labels_quarter_ring,
    "wave": split_labels_islands,
}


def make_dataset(target_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], n: int = 5000):
    a = torch.rand(n, 1)
    b = torch.rand(n, 1)
    y = target_fn(a, b)
    x = torch.cat([a, b], dim=1)
    return x, y


def plot_split_gallery(grid_size: int = 170):
    """Show the 3 requested split charts: original first, then two complex patterns."""
    order = ["balanced", "curved", "wave"]
    a_vals = torch.linspace(0, 1, grid_size)
    b_vals = torch.linspace(0, 1, grid_size)
    AA, BB = torch.meshgrid(a_vals, b_vals, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)

    for ax, name in zip(axes, order):
        labels = SCENARIO_LABELS[name](AA, BB).numpy()
        im = ax.imshow(
            labels.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            vmin=0,
            vmax=2,
            cmap="viridis",
        )
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        if name == "balanced":
            ax.set_title("1) Original split: x+y>1 -> min, else max")
        elif name == "curved":
            ax.set_title("2) Quarter-ring: max(inner) / mean(ring) / min(outer)")
        else:
            ax.set_title("3) Island split + wavy pockets")

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.95, pad=0.02)
    cbar.set_label("Split anchor (0=min, 1=mean/avg, 2=max)")
    fig.suptitle("Input-dependent split patterns (AIGCD target routing)", y=1.02)
    plt.show()


def plot_results_gallery(
    models: dict[str, GenericGLAggregator],
    target_fns: dict[str, Callable],
    anchors: tuple[str, ...],
    grid_size: int = 150,
):
    """3x3 grid: targets (row 1), predictions+anchor overlay (row 2), errors (row 3)."""
    order = ["balanced", "curved", "wave"]
    a_vals = torch.linspace(0, 1, grid_size)
    b_vals = torch.linspace(0, 1, grid_size)
    AA, BB = torch.meshgrid(a_vals, b_vals, indexing="ij")
    x_grid = torch.stack([AA.reshape(-1), BB.reshape(-1)], dim=1)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)

    for col_idx, name in enumerate(order):
        model = models[name]
        target_fn = target_fns[name]

        with torch.no_grad():
            y_pred, alpha, _ = predict_with_details(model, x_grid)
        y_true = target_fn(x_grid[:, 0:1], x_grid[:, 1:2])

        y_true_2d = y_true.reshape(grid_size, grid_size).numpy()
        y_pred_2d = y_pred.reshape(grid_size, grid_size).numpy()
        err_2d = np.abs(y_pred_2d - y_true_2d)

        # Compute effective dominant anchor map (with avg-blend reinterpretation).
        dom_flat = _effective_dominant(alpha.numpy(), anchors)
        dom_2d = dom_flat.reshape(grid_size, grid_size)
        k = len(anchors)
        anchor_colors = ["#3a86ff", "#2ec4b6", "#06d6a0", "#ffd166", "#ef476f", "#9b5de5"]
        dom_cmap = mcolors.ListedColormap(anchor_colors[:k])
        xy = np.linspace(0, 1, grid_size)

        # Row 0: targets
        im0 = axes[0, col_idx].imshow(
            y_true_2d.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        axes[0, col_idx].set_xlabel("a")
        axes[0, col_idx].set_ylabel("b")
        axes[0, col_idx].set_title(f"{col_idx + 1}) {name.capitalize()} target")
        plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.02)

        # Row 1: predictions + dominant anchor region overlay
        ax1 = axes[1, col_idx]
        im1 = ax1.imshow(
            y_pred_2d.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        # Overlay dominant anchor regions (semi-transparent fill).
        ax1.contourf(
            xy, xy, dom_2d.T,
            levels=np.arange(-0.5, k, 1.0),
            colors=anchor_colors[:k],
            alpha=0.25,
        )
        # Draw hard boundary lines between anchor regions.
        ax1.contour(
            xy, xy, dom_2d.T.astype(float),
            levels=np.arange(0.5, k - 0.5, 1.0),
            colors="white",
            linewidths=1.0,
            alpha=0.85,
        )
        # Legend patches for anchor labels.
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(color=anchor_colors[i], alpha=0.7, label=anchors[i])
            for i in range(k)
        ]
        ax1.legend(
            handles=patches, loc="lower right", fontsize=6,
            framealpha=0.6, handlelength=1.0,
        )
        ax1.set_xlabel("a")
        ax1.set_ylabel("b")
        ax1.set_title("AIGCD prediction + anchor regions")

        # Row 2: errors
        im2 = axes[2, col_idx].imshow(
            err_2d.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap="hot",
        )
        axes[2, col_idx].set_xlabel("a")
        axes[2, col_idx].set_ylabel("b")
        axes[2, col_idx].set_title(f"Absolute error (MAE={np.mean(err_2d):.4f})")
        plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.02)

    fig.suptitle("AIGCD value-dependent aggregator: targets, predictions, errors", y=1.02)
    plt.show()


def _effective_dominant(alpha_np: np.ndarray, anchors: tuple[str, ...]) -> np.ndarray:
    """
    Compute dominant anchor index per pixel, with an avg-blend reinterpretation:
    when |w_min - w_max| < blend_tol, the effective operator is arithmetic mean
    (equal mixture of min and max = avg), so relabel that pixel as 'mean'.
    """
    anchors_list = list(anchors)
    dom = np.argmax(alpha_np, axis=1)
    try:
        min_idx = anchors_list.index("min")
        max_idx = anchors_list.index("max")
        mean_idx = anchors_list.index("mean")
    except ValueError:
        return dom  # can't reinterpret without all three present
    w_min = alpha_np[:, min_idx]
    w_max = alpha_np[:, max_idx]
    balanced = np.abs(w_min - w_max) < 0.15
    dom_reinterp = dom.copy()
    dom_reinterp[balanced] = mean_idx
    return dom_reinterp


def predict_with_details(model: GenericGLAggregator, x_batch: torch.Tensor):
    # GenericGL expects [N_inputs, batch]. Here N_inputs=2.
    x_nary = x_batch.T
    u = model.transform(x_nary) if model.use_transform else x_nary
    ops = model._compute_anchors(u)      # [k, batch]
    w = model._compute_weights(u)        # [k, batch]
    y = (w * ops).sum(dim=0)             # [batch]
    alpha = w.T                          # [batch, k]
    return y.unsqueeze(1), alpha, ops.T  # [batch,1], [batch,k], [batch,k]


def train_model(
    model: GenericGLAggregator,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 500,
    lr: float = 1e-2,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_hat, alpha, _ = predict_with_details(model, x_train)
        mse = torch.mean((y_hat - y_train) ** 2)

        # Encourage crisp anchor usage without forcing hard routing too early.
        entropy = -(alpha * torch.log(alpha + EPS)).sum(dim=1).mean()
        loss = mse + 2e-4 * entropy

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_hat, _, _ = predict_with_details(model, x_val)
            val_loss = torch.mean((y_val_hat - y_val) ** 2)

        history["train"].append(float(mse.item()))
        history["val"].append(float(val_loss.item()))

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Train MSE: {mse.item():.6f} | "
                f"Val MSE: {val_loss.item():.6f}"
            )

    return history


def evaluate(model: GenericGLAggregator, x: torch.Tensor, y: torch.Tensor, split_name: str = "Test"):
    model.eval()
    with torch.no_grad():
        y_hat, alpha, _ = predict_with_details(model, x)
        mse = torch.mean((y_hat - y) ** 2).item()
        mae = torch.mean(torch.abs(y_hat - y)).item()

    print(f"\n{split_name} metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    return y_hat, alpha


def plot_training(history: dict, title_prefix: str):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(f"{title_prefix}: training history")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_quality(model: GenericGLAggregator, target_fn, title_prefix: str, n: int = 400):
    model.eval()
    with torch.no_grad():
        x_vis, y_vis = make_dataset(target_fn, n=n)
        y_hat, _, _ = predict_with_details(model, x_vis)

    y_true = y_vis[:, 0].numpy()
    y_pred = y_hat[:, 0].numpy()

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_pred, s=14, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title(f"{title_prefix}: prediction quality")
    plt.tight_layout()
    plt.show()


def plot_maps(
    model: GenericGLAggregator,
    target_fn,
    title_prefix: str,
    anchors: tuple[str, ...],
    grid_size: int = 120,
):
    model.eval()
    a_vals = torch.linspace(0, 1, grid_size)
    b_vals = torch.linspace(0, 1, grid_size)
    AA, BB = torch.meshgrid(a_vals, b_vals, indexing="ij")

    x_grid = torch.stack([AA.reshape(-1), BB.reshape(-1)], dim=1)

    with torch.no_grad():
        y_pred, alpha, _ = predict_with_details(model, x_grid)

    y_true = target_fn(x_grid[:, 0:1], x_grid[:, 1:2])

    y_true_2d = y_true.reshape(grid_size, grid_size).numpy()
    y_pred_2d = y_pred.reshape(grid_size, grid_size).numpy()
    err_2d = np.abs(y_pred_2d - y_true_2d)
    dominant = np.argmax(alpha.numpy(), axis=1).reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    im0 = axes[0].imshow(y_true_2d.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    axes[0].set_title("Target map")
    axes[0].set_xlabel("a")
    axes[0].set_ylabel("b")
    plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.03)

    im1 = axes[1].imshow(y_pred_2d.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    axes[1].set_title("AIGCD prediction")
    axes[1].set_xlabel("a")
    axes[1].set_ylabel("b")
    plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.03)

    im2 = axes[2].imshow(err_2d.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="magma")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("a")
    axes[2].set_ylabel("b")
    plt.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.03)

    im3 = axes[3].imshow(dominant.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="tab10")
    axes[3].set_title("Dominant anchor index")
    axes[3].set_xlabel("a")
    axes[3].set_ylabel("b")
    plt.colorbar(im3, ax=axes[3], fraction=0.045, pad=0.03)

    fig.suptitle(f"{title_prefix} | anchors={anchors}", y=1.02)
    plt.tight_layout()
    plt.show()


def run_scenario(
    name: str,
    epochs: int,
    lr: float,
    n: int,
    hidden_dim: int,
    tau: float,
    anchors: tuple[str, ...],
):
    desc, target_fn = SCENARIOS[name]
    print("=" * 80)
    print(f"Scenario: {name} | {desc}")

    x, y = make_dataset(target_fn, n=n)
    idx = torch.randperm(n)
    x, y = x[idx], y[idx]

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

    model = GenericGLAggregator(
        anchors=anchors,
        weight_mode="value_dependent",
        use_transform=True,
        hidden_dim=hidden_dim,
        tau=tau,
    )

    history = train_model(model, x_train, y_train, x_val, y_val, epochs=epochs, lr=lr)
    _, alpha_test = evaluate(model, x_test, y_test, split_name="Test")

    mean_alpha = alpha_test.mean(dim=0).numpy()
    print("Mean test anchor weights:")
    for anchor_name, w in zip(anchors, mean_alpha):
        print(f"  {anchor_name:>9s}: {w:.4f}")

    plot_training(history, f"{name}")
    plot_quality(model, target_fn, f"{name}")
    plot_maps(model, target_fn, f"{name}", anchors=anchors)
    return model


def run_all_and_gallery(
    epochs: int,
    lr: float,
    n: int,
    hidden_dim: int,
    tau: float,
    anchors: tuple[str, ...],
):
    """Run all three scenarios and show a unified 3x3 results gallery."""
    models = {}
    for name in ("balanced", "curved", "wave"):
        models[name] = run_scenario(
            name=name,
            epochs=epochs,
            lr=lr,
            n=n,
            hidden_dim=hidden_dim,
            tau=tau,
            anchors=anchors,
        )

    target_fns = {name: fn for name, (_, fn) in SCENARIOS.items()}
    plot_results_gallery(models, target_fns, anchors=anchors)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "AIGCD value-dependent aggregator demo using GenericGLAggregator. "
            "Includes baseline plus two more complex input-dependent maps."
        )
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "balanced", "curved", "wave"],
        help="Scenario to run",
    )
    parser.add_argument("--epochs", type=int, default=450, help="Training epochs per scenario")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--n", type=int, default=6000, help="Dataset size per scenario")
    parser.add_argument("--hidden-dim", type=int, default=24, help="Gate network hidden dim")
    parser.add_argument("--tau", type=float, default=0.35, help="Softmax temperature")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--anchors",
        type=str,
        default="simple",
        choices=list(ANCHOR_PRESETS.keys()),
        help="Anchor set to use: simple, extended, or full",
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Only show the 3 split charts (no training/evaluation)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    selected_anchors = ANCHOR_PRESETS[args.anchors]
    print(f"Using anchor set '{args.anchors}': {selected_anchors}")

    # Always show the requested three split charts first.
    plot_split_gallery()

    if args.splits_only:
        return

    if args.scenario == "all":
        run_all_and_gallery(
            epochs=args.epochs,
            lr=args.lr,
            n=args.n,
            hidden_dim=args.hidden_dim,
            tau=args.tau,
            anchors=selected_anchors,
        )
    else:
        run_scenario(
            name=args.scenario,
            epochs=args.epochs,
            lr=args.lr,
            n=args.n,
            hidden_dim=args.hidden_dim,
            tau=args.tau,
            anchors=selected_anchors,
        )


if __name__ == "__main__":
    main()
