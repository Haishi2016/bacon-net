# minimal_conditional_aggregator.py
#
# Self-contained demo of a minimal conditional aggregator.
# Requires: torch, matplotlib
#
# What it does:
# - Generates synthetic data (a, b, c)
# - Learns y = conditional aggregation of a and b based on context c
# - Uses fixed anchor operators: min, mean, max
# - Learns context-dependent mixture weights alpha(c)
#
# Ground-truth rule in this demo:
#   if c < 0.5: y = min(a, b)
#   else:       y = max(a, b)
#
# You can later replace the target rule with other conditional logic.

import math
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
def make_dataset(n=2000):
    # Inputs in [0,1]
    a = torch.rand(n, 1)
    b = torch.rand(n, 1)
    c = torch.rand(n, 1)  # context

    # Ground-truth conditional rule
    y = torch.where(c < 0.5, torch.minimum(a, b), torch.maximum(a, b))

    # Optional small noise
    # y = torch.clamp(y + 0.01 * torch.randn_like(y), 0.0, 1.0)

    x = torch.cat([a, b, c], dim=1)
    return x, y


# -----------------------------
# Conditional aggregator model
# -----------------------------
class ConditionalAggregator(nn.Module):
    """
    Minimal conditional aggregator:
      y_hat = sum_i alpha_i(context) * G_i(a, b)

    Anchors:
      G_1 = min(a,b)
      G_2 = mean(a,b)
      G_3 = max(a,b)

    The gating network produces alpha(context), with softmax normalization.
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def anchors(self, a, b):
        g_min = torch.minimum(a, b)
        g_mean = 0.5 * (a + b)
        g_max = torch.maximum(a, b)
        return torch.cat([g_min, g_mean, g_max], dim=1)

    def forward(self, x):
        a = x[:, 0:1]
        b = x[:, 1:2]
        c = x[:, 2:3]

        anchor_vals = self.anchors(a, b)              # [N, 3]
        logits = self.gate(c)                         # [N, 3]
        alpha = torch.softmax(logits, dim=1)          # [N, 3]

        y_hat = torch.sum(alpha * anchor_vals, dim=1, keepdim=True)
        return y_hat, alpha, anchor_vals


# -----------------------------
# Training
# -----------------------------
def train_model(model, x_train, y_train, x_val, y_val, epochs=1000, lr=1e-2):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_hat, alpha, _ = model(x_train)
        loss = loss_fn(y_hat, y_train)

        # Optional entropy penalty to encourage sharper anchor selection
        # entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=1).mean()
        # loss = loss + 0.001 * entropy

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_hat, _, _ = model(x_val)
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
# Evaluation helpers
# -----------------------------
def evaluate(model, x, y, split_name="Test"):
    model.eval()
    with torch.no_grad():
        y_hat, alpha, _ = model(x)
        mse = torch.mean((y_hat - y) ** 2).item()
        mae = torch.mean(torch.abs(y_hat - y)).item()

    print(f"\n{split_name} metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    return y_hat, alpha


def inspect_gating(model):
    model.eval()
    with torch.no_grad():
        c_grid = torch.linspace(0, 1, 200).unsqueeze(1)
        logits = model.gate(c_grid)
        alpha = torch.softmax(logits, dim=1)

    c_np = c_grid.squeeze().numpy()
    alpha_np = alpha.numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(c_np, alpha_np[:, 0], label="alpha_min")
    plt.plot(c_np, alpha_np[:, 1], label="alpha_mean")
    plt.plot(c_np, alpha_np[:, 2], label="alpha_max")
    plt.axvline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("context c")
    plt.ylabel("anchor weight")
    plt.title("Learned conditional anchor weights")
    plt.legend()
    plt.tight_layout()
    plt.show()


def inspect_predictions(model, n=300):
    model.eval()
    with torch.no_grad():
        x_vis, y_vis = make_dataset(n)
        y_hat, alpha, _ = model(x_vis)

    a = x_vis[:, 0].numpy()
    b = x_vis[:, 1].numpy()
    c = x_vis[:, 2].numpy()
    y_true = y_vis[:, 0].numpy()
    y_pred = y_hat[:, 0].numpy()
    alpha_np = alpha.numpy()

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

    # Show a few examples
    print("\nSample predictions:")
    print("   a      b      c    y_true  y_pred   alpha[min,mean,max]")
    for i in range(10):
        print(
            f"{a[i]:.3f}  {b[i]:.3f}  {c[i]:.3f}   "
            f"{y_true[i]:.3f}   {y_pred[i]:.3f}   "
            f"[{alpha_np[i,0]:.3f}, {alpha_np[i,1]:.3f}, {alpha_np[i,2]:.3f}]"
        )


# -----------------------------
# Main
# -----------------------------
def main():
    # Data
    x, y = make_dataset(3000)

    # Split
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

    # Model
    model = ConditionalAggregator(hidden=16)

    # Train
    history = train_model(
        model,
        x_train, y_train,
        x_val, y_val,
        epochs=1000,
        lr=1e-2
    )

    # Evaluate
    evaluate(model, x_test, y_test, split_name="Test")

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Inspect learned conditional behavior
    inspect_gating(model)
    inspect_predictions(model, n=300)


if __name__ == "__main__":
    main()