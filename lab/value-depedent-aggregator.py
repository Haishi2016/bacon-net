# value_dependent_aggregator_balanced.py
#
# Self-contained demo of a value-dependent aggregator with an easier distribution.
# Requires: torch, matplotlib
#
# Ground-truth rule:
#   if a + b > 1.0:
#       y = min(a,b)
#   else:
#       y = max(a,b)
#
# This is easier than the "both > 0.7" rule because the two regimes are
# much more balanced and the decision boundary occupies a large region.

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# Synthetic dataset
# -----------------------------
def make_dataset(n=4000):
    a = torch.rand(n, 1)
    b = torch.rand(n, 1)

    condition = (a + b > 1.0)
    y = torch.where(condition, torch.minimum(a, b), torch.maximum(a, b))

    x = torch.cat([a, b], dim=1)
    return x, y


# -----------------------------
# Value-dependent aggregator
# -----------------------------
class ValueDependentAggregator(nn.Module):
    """
    y_hat = sum_i alpha_i(gate_features(a,b)) * G_i(a,b)

    Anchors:
      G1 = min(a,b)
      G2 = mean(a,b)
      G3 = max(a,b)

    Gate sees both inputs plus joint features:
      [a, b, a*b, |a-b|, (a+b)/2]
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def anchors(self, a, b):
        g_min = torch.minimum(a, b)
        g_mean = 0.5 * (a + b)
        g_max = torch.maximum(a, b)
        return torch.cat([g_min, g_mean, g_max], dim=1)

    def gate_features(self, a, b):
        ab = a * b
        diff = torch.abs(a - b)
        mean = 0.5 * (a + b)
        return torch.cat([a, b, ab, diff, mean], dim=1)

    def forward(self, x):
        a = x[:, 0:1]
        b = x[:, 1:2]

        anchor_vals = self.anchors(a, b)              # [N, 3]
        gate_input = self.gate_features(a, b)         # [N, 5]
        logits = self.gate(gate_input)                # [N, 3]
        alpha = torch.softmax(logits, dim=1)          # [N, 3]

        y_hat = torch.sum(alpha * anchor_vals, dim=1, keepdim=True)
        return y_hat, alpha, anchor_vals, gate_input


# -----------------------------
# Training
# -----------------------------
def train_model(model, x_train, y_train, x_val, y_val, epochs=800, lr=1e-2):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_hat, alpha, _, _ = model(x_train)
        loss = loss_fn(y_hat, y_train)

        # Optional sharpness regularization:
        # entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=1).mean()
        # loss = loss + 0.0002 * entropy

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_hat, _, _, _ = model(x_val)
            val_loss = loss_fn(val_hat, y_val)

        history["train"].append(loss.item())
        history["val"].append(val_loss.item())

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {loss.item():.6f} | "
                f"Val Loss: {val_loss.item():.6f}"
            )

    return history


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, x, y, split_name="Test"):
    model.eval()
    with torch.no_grad():
        y_hat, alpha, _, _ = model(x)
        mse = torch.mean((y_hat - y) ** 2).item()
        mae = torch.mean(torch.abs(y_hat - y)).item()

    print(f"\n{split_name} metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    return y_hat, alpha


# -----------------------------
# Visualization helpers
# -----------------------------
def plot_training(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_quality(model, n=400):
    model.eval()
    with torch.no_grad():
        x_vis, y_vis = make_dataset(n)
        y_hat, _, _, _ = model(x_vis)

    y_true = y_vis[:, 0].numpy()
    y_pred = y_hat[:, 0].numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Prediction quality")
    plt.tight_layout()
    plt.show()


def plot_anchor_regions(model, grid_size=120):
    """
    Visualize which anchor dominates over the (a,b) plane.
    """
    model.eval()

    a_vals = torch.linspace(0, 1, grid_size)
    b_vals = torch.linspace(0, 1, grid_size)

    AA, BB = torch.meshgrid(a_vals, b_vals, indexing="ij")
    x_grid = torch.stack([AA.reshape(-1), BB.reshape(-1)], dim=1)

    with torch.no_grad():
        _, alpha, _, _ = model(x_grid)

    alpha = alpha.numpy()
    dominant = np.argmax(alpha, axis=1).reshape(grid_size, grid_size)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        dominant.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto"
    )
    plt.colorbar(label="Dominant anchor (0=min, 1=mean, 2=max)")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.title("Dominant learned anchor over input space")
    plt.tight_layout()
    plt.show()


def inspect_samples(model, n=20):
    model.eval()
    with torch.no_grad():
        x_vis, y_vis = make_dataset(n)
        y_hat, alpha, _, gate_input = model(x_vis)

    x_np = x_vis.numpy()
    y_true = y_vis[:, 0].numpy()
    y_pred = y_hat[:, 0].numpy()
    alpha_np = alpha.numpy()
    gate_np = gate_input.numpy()

    print("\nSample predictions:")
    print("   a      b    y_true  y_pred   alpha[min,mean,max]      gate_features[a,b,ab,|a-b|,mean]")
    for i in range(n):
        a, b = x_np[i]
        gf = gate_np[i]
        print(
            f"{a:.3f}  {b:.3f}   "
            f"{y_true[i]:.3f}   {y_pred[i]:.3f}   "
            f"[{alpha_np[i,0]:.3f}, {alpha_np[i,1]:.3f}, {alpha_np[i,2]:.3f}]   "
            f"[{gf[0]:.3f}, {gf[1]:.3f}, {gf[2]:.3f}, {gf[3]:.3f}, {gf[4]:.3f}]"
        )


# -----------------------------
# Main
# -----------------------------
def main():
    x, y = make_dataset(5000)

    n = x.shape[0]
    idx = torch.randperm(n)
    x = x[idx]
    y = y[idx]

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_val = x[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    x_test = x[n_train + n_val:]
    y_test = y[n_train + n_val:]

    model = ValueDependentAggregator(hidden=16)

    history = train_model(
        model,
        x_train, y_train,
        x_val, y_val,
        epochs=800,
        lr=1e-2
    )

    evaluate(model, x_test, y_test, split_name="Test")
    plot_training(history)
    plot_prediction_quality(model)
    plot_anchor_regions(model)
    inspect_samples(model, n=20)


if __name__ == "__main__":
    main()