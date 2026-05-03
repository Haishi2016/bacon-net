"""conditional-aggregator-full-aigcd.py

Test of AIGCD mode with ALL components enabled simultaneously:
  - Routing logits (static anchor routing)
  - Gating network (context-dependent)
  - Value-dependent gating (input feature-based adjustment)
  - Coordinate transformation R

Tests whether the model can learn the conditional two-band target
(min when c<0.5, max when c>=0.5) while juggling all components.
"""

import pathlib
import random
import sys

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
ANCHORS = ("min", "mean", "max")
C_SWITCH = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---- Dataset ----

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


# ---- Forward pass ----

def predict(
    model: GenericGLAggregator,
    x_ab: torch.Tensor,
    c: torch.Tensor,
):
    """Run forward pass and return (y_hat [N,1], alpha [N,k], R)."""
    x_nary = x_ab.T                                  # [2, N]
    u = model.transform(x_nary) if model.use_transform else x_nary
    ops = model._compute_anchors(u)                  # [k, N]
    w   = model._compute_weights(u, c=c)             # [k, N]
    y   = (w * ops).sum(dim=0, keepdim=True).T       # [N, 1]
    alpha = w.T                                      # [N, k]
    R = model.get_transform_matrix() if model.use_transform else None
    return y, alpha, R


# ---- Training ----

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

        y_hat, alpha, _ = predict(model, x_train, c_train)
        mse = torch.mean((y_hat - y_train) ** 2)

        # Light entropy penalty + transform regularization (identity prior).
        entropy = -(alpha * torch.log(alpha + EPS)).sum(dim=1).mean()
        reg = model.transform_regularization()
        loss = mse + 1e-4 * entropy + reg

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_hat, _, _ = predict(model, x_val, c_val)
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


# ---- Evaluation ----

def evaluate(
    model: GenericGLAggregator,
    x: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    split_name: str = "Test",
):
    model.eval()
    with torch.no_grad():
        y_hat, alpha, R = predict(model, x, c)
        mse = torch.mean((y_hat - y) ** 2).item()
        mae = torch.mean(torch.abs(y_hat - y)).item()

    print(f"\n{split_name} metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    if R is not None:
        print(f"\nTransform matrix R (first 2x2, identity-initialized):")
        print(R.detach().cpu().numpy())
    
    return y_hat, alpha


# ---- Plots ----

def plot_gating_bands(model: GenericGLAggregator, anchors: tuple[str, ...]):
    """Show learned anchor weights vs context c (with all components active)."""
    model.eval()
    with torch.no_grad():
        c_grid = torch.linspace(0, 1, 400).unsqueeze(1)   # [400, 1]
        # Use representative input values (e.g., center of [0,1]^2).
        x_dummy = torch.full((400, 2), 0.5)
        _, alpha, _ = predict(model, x_dummy, c_grid)

    c_np = c_grid.squeeze().numpy()
    alpha_np = alpha.numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3a86ff", "#2ec4b6", "#ef476f"]
    for i, (name, col) in enumerate(zip(anchors, colors)):
        ax.plot(c_np, alpha_np[:, i], label=f"alpha_{name}", color=col, linewidth=2)

    ax.axvline(C_SWITCH, linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.axvspan(0.0, C_SWITCH, alpha=0.06, color="#3a86ff", label=f"min zone (c<{C_SWITCH})")
    ax.axvspan(C_SWITCH, 1.0, alpha=0.06, color="#ef476f", label=f"max zone (c>={C_SWITCH})")

    ax.set_xlabel("context c")
    ax.set_ylabel("anchor weight")
    ax.set_title("AIGCD Full Mode: Learned anchor weights α(c) with all components active")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    plt.show()


def plot_training(history: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"],   label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("AIGCD Full Mode: Training history (conditional target)")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---- Main ----

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

    # ---- Model: ALL AIGCD components enabled ----
    model = GenericGLAggregator(
        anchors=ANCHORS,
        weight_mode="aigcd",
        use_routing=True,                  # static routing logits
        use_gating=True,                   # gating network
        gating_use_values=True,            # input-dependent gating
        gating_use_context=True,           # context-dependent gating
        use_transform=True,                # coordinate transformation R
        context_dim=1,
        hidden_dim=24,
        tau=1.0,
        identity_reg=0.01,                 # light regularization toward identity R
    )
    print(model)
    print(f"\nTwo-band conditional target: min(c<{C_SWITCH})  max(c>={C_SWITCH})")
    print("All AIGCD components active: routing + gating(context+values) + transform R\n")

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

    # ---- Diagnostics ----
    # For AIGCD with full components, we need to provide u and c for describe()
    with torch.no_grad():
        u_sample = torch.full((2, 1), 0.5)  # representative input [N=2, batch=1]
        c_sample = torch.tensor([[0.5]])     # context at switching point
        info = model.describe(u=u_sample, c=c_sample)
    print("\nModel diagnostics (at u=0.5, c=0.5):")
    print(f"  Effective andness: {info['effective_andness']:.4f}")
    print(f"  Dominant anchor: {info['dominant_op']}")
    print(f"  Anchor weights: {', '.join(f'{n}={w:.4f}' for n, w in zip(info['anchors'], info['weights']))}")

    # ---- Plots ----
    plot_training(history)
    plot_gating_bands(model, ANCHORS)


if __name__ == "__main__":
    main()
