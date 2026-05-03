import argparse
import pathlib
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure local repo package imports work when running this script directly.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bacon.aggregators.lsp.generic_gl import GenericGLAggregator, ANCHOR_ANDNESS

EPS = 1e-9
ANCHORS = ("min", "harmonic", "geometric", "mean", "quadratic", "max")


def _row_probs_to_logits(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return np.log(p)


def _set_transform_for_cpa(agg: GenericGLAggregator, w_eff: float) -> None:
    # u1 = x, u2 = A_w(x,y) = w_eff*x + (1-w_eff)*y
    row1 = _row_probs_to_logits(np.array([1.0, 0.0]))
    row2 = _row_probs_to_logits(np.array([w_eff, 1.0 - w_eff]))
    logits = torch.tensor(np.vstack([row1, row2]), dtype=torch.float32)
    with torch.no_grad():
        agg.r_logits.copy_(logits)


def _set_anchor_probs(agg: GenericGLAggregator, probs: np.ndarray) -> None:
    logits = torch.tensor(_row_probs_to_logits(probs), dtype=torch.float32)
    with torch.no_grad():
        agg.alpha_logits.copy_(logits)


def _build_cpa_agg(
    probs: np.ndarray,
    w: float,
    gamma: float = 1.0,
    tau: float = 1e-3,
) -> GenericGLAggregator:
    agg = GenericGLAggregator(
        anchors=ANCHORS,
        # AIGCD mode keeps routing + gating + transform available together.
        weight_mode="aigcd",
        use_transform=True,
        use_routing=True,
        use_gating=True,
        gating_use_values=True,
        gating_use_context=False,
        tau=tau,
    )
    _set_anchor_probs(agg, probs)
    w_eff = float(np.clip(w, EPS, 1.0 - EPS) ** gamma)
    _set_transform_for_cpa(agg, w_eff)
    return agg


def _eval_pair(agg: GenericGLAggregator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    inp = torch.stack([xt, yt], dim=0)
    with torch.no_grad():
        out = agg(inp)
    return out.cpu().numpy()


def model_penalty_rel(probs: np.ndarray, W: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    vals = np.zeros_like(W)
    for i, w in enumerate(W):
        agg = _build_cpa_agg(probs, float(w), gamma=gamma)
        out = _eval_pair(agg, np.array([1.0]), np.array([0.0]))[0]
        vals[i] = 100.0 * (out - 1.0)
    return vals


def model_reward_rel(probs: np.ndarray, x: float, W: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    vals = np.zeros_like(W)
    for i, w in enumerate(W):
        agg = _build_cpa_agg(probs, float(w), gamma=gamma)
        out = _eval_pair(agg, np.array([x]), np.array([1.0]))[0]
        vals[i] = 100.0 * (out / max(x, EPS) - 1.0)
    return vals


def analytic_ag_penalty_rel(W: np.ndarray) -> np.ndarray:
    return -100.0 * (1.0 - np.sqrt(W))


def analytic_ag_reward_rel(x: float, W: np.ndarray) -> np.ndarray:
    return 100.0 * (np.sqrt(W + (1.0 - W) / x) - 1.0)


def analytic_ah_penalty_rel(W: np.ndarray) -> np.ndarray:
    return -100.0 * ((1.0 - W) / (1.0 + W))


def analytic_ah_reward_rel(x: float, W: np.ndarray) -> np.ndarray:
    return 100.0 * ((1.0 - W) * (1.0 - x)) / (x * (W + 1.0) + (1.0 - W))


def _anchors_on_pair_np(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_safe = np.clip(x, EPS, None)
    y_safe = np.clip(y, EPS, None)

    return {
        "min": np.minimum(x, y),
        "harmonic": 2.0 / (1.0 / x_safe + 1.0 / y_safe),
        "geometric": np.sqrt(x_safe * y_safe),
        "mean": 0.5 * (x + y),
        "quadratic": np.sqrt(0.5 * (x * x + y * y)),
        "max": np.maximum(x, y),
    }


def _fast_model_pair(probs: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, gamma: float) -> np.ndarray:
    w_eff = np.clip(w, EPS, 1.0 - EPS) ** gamma
    aw = w_eff * x + (1.0 - w_eff) * y
    anchors = _anchors_on_pair_np(x, aw)
    out = np.zeros_like(w_eff, dtype=np.float64)
    for i, name in enumerate(ANCHORS):
        out = out + probs[i] * anchors[name]
    return out


def _fast_curves(probs: np.ndarray, W: np.ndarray, xs: list[float], gamma: float) -> Tuple[np.ndarray, list[np.ndarray]]:
    penalty = 100.0 * (_fast_model_pair(probs, np.ones_like(W), np.zeros_like(W), W, gamma) - 1.0)
    rewards = []
    for x in xs:
        xx = np.full_like(W, x, dtype=np.float64)
        yy = np.ones_like(W, dtype=np.float64)
        out = _fast_model_pair(probs, xx, yy, W, gamma)
        rewards.append(100.0 * (out / max(x, EPS) - 1.0))
    return penalty, rewards


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def search_hybrid_profile(W: np.ndarray, xs: list[float], n_samples: int = 2000, seed: int = 42) -> Tuple[np.ndarray, float, dict]:
    rng = np.random.default_rng(seed)

    # Targets: penalty should look AH-like, rewards should look AG-like.
    target_pen = analytic_ah_penalty_rel(W)
    target_rews = [analytic_ag_reward_rel(x, W) for x in xs]

    best = {
        "score": float("inf"),
        "probs": None,
        "gamma": None,
        "pen": None,
        "rews": None,
    }

    gammas = np.linspace(0.7, 1.7, 17)

    for gamma in gammas:
        for _ in range(n_samples // len(gammas)):
            probs = rng.dirichlet(np.array([0.4, 2.0, 2.0, 1.0, 0.6, 0.4]))
            pen, rews = _fast_curves(probs, W, xs, float(gamma))
            score = _rmse(pen, target_pen)
            for pred, target in zip(rews, target_rews):
                score += _rmse(pred, target)
            if score < best["score"]:
                best.update({
                    "score": score,
                    "probs": probs,
                    "gamma": float(gamma),
                    "pen": pen,
                    "rews": rews,
                })

    metrics = {
        "penalty_rmse_to_ah": _rmse(best["pen"], target_pen),
        "penalty_rmse_to_ag": _rmse(best["pen"], analytic_ag_penalty_rel(W)),
        "reward_rmse_to_ag": float(np.mean([_rmse(rp, rt) for rp, rt in zip(best["rews"], target_rews)])),
        "reward_rmse_to_ah": float(np.mean([_rmse(rp, rt) for rp, rt in zip(best["rews"], [analytic_ah_reward_rel(x, W) for x in xs])])),
    }
    return best["probs"], best["gamma"], metrics


def inferred_andness(probs: np.ndarray) -> float:
    andness_values = np.array([ANCHOR_ANDNESS[name] for name in ANCHORS], dtype=np.float64)
    return float(np.dot(probs, andness_values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-panel CPA visuals: AG, AH, and hybrid AIGCD profile")
    parser.add_argument("--show", action="store_true", help="Show matplotlib figure")
    parser.add_argument("--save", type=str, default=str(pathlib.Path(__file__).with_suffix(".png")), help="Save figure path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for hybrid-profile search")
    args = parser.parse_args()

    W = np.linspace(0.05, 0.98, 300)
    xs = [0.25, 0.5, 0.75]

    probs_ag = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    probs_ah = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    ag_pen = model_penalty_rel(probs_ag, W)
    ah_pen = model_penalty_rel(probs_ah, W)
    ag_rewards = [model_reward_rel(probs_ag, x, W) for x in xs]
    ah_rewards = [model_reward_rel(probs_ah, x, W) for x in xs]

    ag_pen_ref = analytic_ag_penalty_rel(W)
    ah_pen_ref = analytic_ah_penalty_rel(W)
    ag_rewards_ref = [analytic_ag_reward_rel(x, W) for x in xs]
    ah_rewards_ref = [analytic_ah_reward_rel(x, W) for x in xs]

    probs_hybrid, gamma_hybrid, hybrid_metrics = search_hybrid_profile(W, xs, seed=args.seed)
    hybrid_pen = model_penalty_rel(probs_hybrid, W, gamma=gamma_hybrid)
    hybrid_rewards = [model_reward_rel(probs_hybrid, x, W, gamma=gamma_hybrid) for x in xs]

    colors = {
        "penalty": "black",
        0.25: "tab:blue",
        0.5: "tab:orange",
        0.75: "tab:green",
    }

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    # Panel 1: AG equivalence
    ax = axes[0]
    for x, y in zip(xs, ag_rewards_ref):
        ax.plot(W, y, linewidth=2, color=colors[x], label=f"Reward x={x} (analytic AG)")
    ax.plot(W, ag_pen_ref, linewidth=2, color=colors["penalty"], label="Penalty (analytic AG)")

    stride = 6
    for x, y in zip(xs, ag_rewards):
        ax.plot(W[::stride], y[::stride], linestyle="None", marker="o", markersize=3, color=colors[x], label=f"Reward x={x} (AIGCD)")
    ax.plot(W[::stride], ag_pen[::stride], linestyle="None", marker="o", markersize=3, color=colors["penalty"], label="Penalty (AIGCD)")

    ax.set_title("AG CPA: analytic lines + AIGCD dots")
    ax.set_xlabel("Weight W1")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Penalty [-%] and reward [+%]")
    ax.set_xlim(0, 1)
    ax.set_ylim(-100, 100)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(-100, 110, 10))
    ax.grid(True)

    # Panel 2: AH equivalence
    ax = axes[1]
    for x, y in zip(xs, ah_rewards_ref):
        ax.plot(W, y, linewidth=2, color=colors[x], label=f"Reward x={x} (analytic AH)")
    ax.plot(W, ah_pen_ref, linewidth=2, color=colors["penalty"], label="Penalty (analytic AH)")

    for x, y in zip(xs, ah_rewards):
        ax.plot(W[::stride], y[::stride], linestyle="None", marker="o", markersize=3, color=colors[x], label=f"Reward x={x} (AIGCD)")
    ax.plot(W[::stride], ah_pen[::stride], linestyle="None", marker="o", markersize=3, color=colors["penalty"], label="Penalty (AIGCD)")

    ax.set_title("AH CPA: analytic lines + AIGCD dots")
    ax.set_xlabel("Weight W1")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Penalty [-%] and reward [+%]")
    ax.set_xlim(0, 1)
    ax.set_ylim(-100, 100)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(-100, 110, 10))
    ax.grid(True)

    # Panel 3: Hybrid profile (AH-like penalty, AG-like rewards)
    ax = axes[2]
    for x, y in zip(xs, ag_rewards_ref):
        ax.plot(W, y, linewidth=1.5, color=colors[x], alpha=0.65, label=f"Reward x={x} AG ref")
    for x, y in zip(xs, ah_rewards_ref):
        ax.plot(W, y, linewidth=1.2, linestyle="--", color=colors[x], alpha=0.65, label=f"Reward x={x} AH ref")

    ax.plot(W, ah_pen_ref, linewidth=1.5, color=colors["penalty"], alpha=0.75, label="Penalty AH ref")
    ax.plot(W, ag_pen_ref, linewidth=1.2, linestyle="--", color="gray", alpha=0.75, label="Penalty AG ref")

    for x, y in zip(xs, hybrid_rewards):
        ax.plot(W[::stride], y[::stride], linestyle="None", marker="o", markersize=3, color=colors[x], label=f"Reward x={x} AIGCD hybrid")
    ax.plot(W[::stride], hybrid_pen[::stride], linestyle="None", marker="o", markersize=3, color=colors["penalty"], label="Penalty AIGCD hybrid")

    ax.set_title("Hybrid AIGCD: AH-like penalty, AG-like rewards")
    ax.set_xlabel("Weight W1")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Penalty [-%] and reward [+%]")
    ax.set_xlim(0, 1)
    ax.set_ylim(-100, 100)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(-100, 110, 10))
    ax.grid(True)

    for ax in axes:
        ax.legend(fontsize=7, ncol=1)

    plt.tight_layout()

    # Report hybrid operator details.
    andness = inferred_andness(probs_hybrid)
    print("\nAIGCD hybrid profile (searched):")
    print(f"gamma={gamma_hybrid:.3f}, effective_andness={andness:.3f}")
    print("anchor_probs:")
    for name, p in zip(ANCHORS, probs_hybrid):
        print(f"  {name:>9s}: {p:.4f}")
    print("fit metrics:")
    for k, v in hybrid_metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.save:
        plt.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"\nSaved figure: {args.save}")

    if args.show or not args.save:
        plt.show()


if __name__ == "__main__":
    main()
