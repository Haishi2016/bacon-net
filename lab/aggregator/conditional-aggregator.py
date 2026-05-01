"""conditional-aggregator.py

Demo of context-dependent aggregation using GenericGLAggregator
(weight_mode='conditional').

Ground-truth rule (two bands):
    c < 0.5           -> y = min(a, b)
    c >= 0.5          -> y = max(a, b)

The model receives (a, b) as aggregation inputs and c as external context.
It learns alpha(c) -- the convex anchor weights -- via a small gate network.
"""

import pathlib
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# Repo import
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bacon.aggregators.lsp.generic_gl import GenericGLAggregator

SEED = 42
EPS = 1e-8
ANCHORS = ("min", "mean", "max")
# Context boundary
C_SWITCH = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(n: int = 3000):
    """Generate (a, b, c) with two-band conditional target."""
    a = torch.rand(n, 1)
    b = torch.rand(n, 1)
    c = torch.rand(n, 1)

    y_min = torch.minimum(a, b)
    y_max = torch.maximum(a, b)

    y = torch.where(c < C_SWITCH, y_min, y_max)

    x_ab = torch.cat([a, b], dim=1)  # aggregation inputs [N, 2]
    return x_ab, c, y


# ---------------------------------------------------------------------------
# Forward helper (uses GenericGL in conditional mode)
# ---------------------------------------------------------------------------

def predict(
    model: GenericGLAggregator,
    x_ab: torch.Tensor,
    c: torch.Tensor,
):
    """Run forward pass and return (y_hat [N,1], alpha [N,k])."""
    # GenericGL forward expects x [N_inputs, batch].
    x_nary = x_ab.T                                  # [2, N]
    u = model.transform(x_nary) if model.use_transform else x_nary
    ops = model._compute_anchors(u)                  # [k, N]
    w   = model._compute_weights(u, c=c)             # [k, N]
    y   = (w * ops).sum(dim=0, keepdim=True).T       # [N, 1]
    alpha = w.T                                      # [N, k]
    return y, alpha


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: GenericGLAggregator,
    x_train: torch.Tensor,
    c_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    c_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 1000,
    lr: float = 1e-2,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_hat, alpha = predict(model, x_train, c_train)
        mse = torch.mean((y_hat - y_train) ** 2)

        # Light entropy penalty for crisper routing.
        entropy = -(alpha * torch.log(alpha + EPS)).sum(dim=1).mean()
        loss = mse + 1e-4 * entropy

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_hat, _ = predict(model, x_val, c_val)
            val_loss = torch.mean((val_hat - y_val) ** 2)

        history["train"].append(float(mse.item()))
        history["val"].append(float(val_loss.item()))

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Train MSE: {mse.item():.6f} | "
                f"Val MSE: {val_loss.item():.6f}"
            )

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: GenericGLAggregator,
    x: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    split_name: str = "Test",
):
    model.eval()
    with torch.no_grad():
        y_hat, alpha = predict(model, x, c)
        mse = torch.mean((y_hat - y) ** 2).item()
        mae = torch.mean(torch.abs(y_hat - y)).item()

    print(f"\n{split_name} metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    return y_hat, alpha


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_gating(model: GenericGLAggregator, anchors: tuple[str, ...]):
    """Reproduce and extend the original chart: anchor weights vs context c.
    Shows two bands (c<0.5 -> min, c>=0.5 -> max).
    """
    model.eval()
    with torch.no_grad():
        c_grid = torch.linspace(0, 1, 400).unsqueeze(1)   # [400, 1]
        # Dummy a=b=0.5 inputs (gating in conditional mode only reads c).
        x_dummy = torch.full((400, 2), 0.5)
        _, alpha = predict(model, x_dummy, c_grid)

    c_np = c_grid.squeeze().numpy()
    alpha_np = alpha.numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3a86ff", "#2ec4b6", "#ef476f"]
    for i, (name, col) in enumerate(zip(anchors, colors)):
        ax.plot(c_np, alpha_np[:, i], label=f"alpha_{name}", color=col, linewidth=2)

    # Band boundary
    ax.axvline(C_SWITCH, linestyle="--", linewidth=1, color="gray", alpha=0.7)

    # Shaded band labels
    ax.axvspan(0.0, C_SWITCH, alpha=0.06, color="#3a86ff", label=f"min zone (c<{C_SWITCH})")
    ax.axvspan(C_SWITCH, 1.0, alpha=0.06, color="#ef476f", label=f"max zone (c>={C_SWITCH})")

    ax.set_xlabel("context c")
    ax.set_ylabel("anchor weight")
    ax.set_title("Learned conditional anchor weights α(c)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    plt.show()


def plot_predictions(model: GenericGLAggregator, n: int = 400):
    """Scatter plot of true vs predicted y."""
    model.eval()
    with torch.no_grad():
        x_vis, c_vis, y_vis = make_dataset(n)
        y_hat, _ = predict(model, x_vis, c_vis)

    y_true = y_vis[:, 0].numpy()
    y_pred = y_hat[:, 0].numpy()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted y")
    ax.set_title("Prediction quality")
    fig.tight_layout()
    plt.show()


def plot_training(history: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"],   label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    # ---- Data ----
    x, c, y = make_dataset(n=4000)
    idx = torch.randperm(len(y))
    x, c, y = x[idx], c[idx], y[idx]

    n = len(y)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    x_train, c_train, y_train = x[:n_train],           c[:n_train],           y[:n_train]
    x_val,   c_val,   y_val   = x[n_train:n_train+n_val], c[n_train:n_train+n_val], y[n_train:n_train+n_val]
    x_test,  c_test,  y_test  = x[n_train+n_val:],    c[n_train+n_val:],    y[n_train+n_val:]

    # ---- Model ----
    model = GenericGLAggregator(
        anchors=ANCHORS,
        weight_mode="conditional",
        context_dim=1,
        use_transform=False,   # no coordinate transform needed; a,b are already clean inputs
        hidden_dim=24,
        tau=1.0,
    )
    print(model)
    print(f"\nTwo-band target: min(c<{C_SWITCH})  max(c>={C_SWITCH})\n")

    # ---- Train ----
    history = train_model(
        model,
        x_train, c_train, y_train,
        x_val,   c_val,   y_val,
        epochs=1200,
        lr=1e-2,
    )

    # ---- Evaluate ----
    evaluate(model, x_test, c_test, y_test, split_name="Test")

    # ---- Plots ----
    plot_training(history)
    plot_gating(model, ANCHORS)
    plot_predictions(model)


if __name__ == "__main__":
    main()
