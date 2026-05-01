import argparse
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure local repo package imports work when running this script directly.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bacon.aggregators.lsp.generic_gl import GenericGLAggregator, ANCHOR_ANDNESS

EPS = 1e-9
ANCHORS = ("min", "harmonic", "geometric", "mean", "quadratic", "max")


def _row_probs_to_logits(p, eps=1e-8):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return np.log(p)


def _set_transform_for_cpa(agg: GenericGLAggregator, w: float) -> None:
    # u1 = x, u2 = A_w(x,y) = w*x + (1-w)*y
    row1 = _row_probs_to_logits([1.0, 0.0])
    row2 = _row_probs_to_logits([w, 1.0 - w])
    logits = torch.tensor(np.vstack([row1, row2]), dtype=torch.float32)
    with torch.no_grad():
        agg.r_logits.copy_(logits)


def _set_anchor_mode(agg: GenericGLAggregator, mode: str) -> None:
    logits = torch.full((len(ANCHORS),), -20.0, dtype=torch.float32)
    if mode == "AG":
        idx = ANCHORS.index("geometric")
    elif mode == "AH":
        idx = ANCHORS.index("harmonic")
    else:
        raise ValueError("mode must be 'AG' or 'AH'")
    logits[idx] = 20.0
    with torch.no_grad():
        agg.alpha_logits.copy_(logits)


def _build_cpa_agg(mode: str, w: float, tau: float = 1e-3) -> GenericGLAggregator:
    agg = GenericGLAggregator(
        anchors=ANCHORS,
        weight_mode="static",
        use_transform=True,
        tau=tau,
    )
    _set_anchor_mode(agg, mode)
    _set_transform_for_cpa(agg, w)
    return agg


def _eval_pair(agg: GenericGLAggregator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    inp = torch.stack([xt, yt], dim=0)  # [2, ...]
    with torch.no_grad():
        out = agg(inp)
    return out.cpu().numpy()


def penalty_rel(mode: str, W: np.ndarray) -> np.ndarray:
    vals = np.zeros_like(W)
    for i, w in enumerate(W):
        agg = _build_cpa_agg(mode, float(w))
        out = _eval_pair(agg, np.array([1.0]), np.array([0.0]))[0]
        vals[i] = 100.0 * (out - 1.0)
    return vals


def reward_rel(mode: str, x: float, W: np.ndarray) -> np.ndarray:
    vals = np.zeros_like(W)
    for i, w in enumerate(W):
        agg = _build_cpa_agg(mode, float(w))
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


def inferred_penalty_reward_from_weights(agg: GenericGLAggregator) -> tuple[float, float, float]:
    """
    Infer behavior from anchor weights (no explicit reward/penalty assignment).

    Returns:
      effective_andness in [~0,~1+],
      penalty_propensity in [0,100],
      reward_propensity in [0,100].
    """
    info = agg.describe()
    w = np.array(info["weights"], dtype=np.float64)
    a = np.array([ANCHOR_ANDNESS[name] for name in info["anchors"]], dtype=np.float64)

    # Andness > 0.5 leans conjunctive (penalty-preserving);
    # andness < 0.5 leans disjunctive (reward-amplifying).
    andness = float(np.dot(w, a))
    penalty_propensity = float(np.clip(2.0 * (andness - 0.5), 0.0, 1.0) * 100.0)
    reward_propensity = float(np.clip(2.0 * (0.5 - andness), 0.0, 1.0) * 100.0)
    return andness, penalty_propensity, reward_propensity


def _print_transform_matrices(w_samples: list[float]) -> None:
    print("\nTransformation matrices R (u = R x):")
    for w in w_samples:
        agg = _build_cpa_agg("AG", w)
        R = agg.get_transform_matrix().detach().cpu().numpy()
        print(f"W={w:.2f} ->")
        print(np.array2string(R, precision=4, suppress_small=True))


def _print_traditional_vs_aigcd_table(xs: list[float], w_samples: list[float]) -> None:
    """One consolidated table comparing analytic (traditional) vs AIGCD values."""
    print("\nTraditional (analytic) vs AIGCD comparison table")
    print("Values are in percent. For reward: 100*(A(x,1)/x - 1). For penalty: 100*(A(1,0)-1).")
    header = (
        f"{'mode':<4} | {'metric':<7} | {'x':>4} | {'W':>4} | "
        f"{'traditional':>11} | {'AIGCD':>9} | {'|diff|':>8}"
    )
    print(header)
    print("-" * len(header))

    for mode in ("AG", "AH"):
        for w in w_samples:
            # penalty row (x fixed at 1, y=0)
            agg = _build_cpa_agg(mode, w)
            out_pen = _eval_pair(agg, np.array([1.0]), np.array([0.0]))[0]
            aigcd_pen = 100.0 * (out_pen - 1.0)
            if mode == "AG":
                trad_pen = float(analytic_ag_penalty_rel(np.array([w]))[0])
            else:
                trad_pen = float(analytic_ah_penalty_rel(np.array([w]))[0])
            print(
                f"{mode:<4} | {'penalty':<7} | {'1.00':>4} | {w:>4.2f} | "
                f"{trad_pen:11.4f} | {aigcd_pen:9.4f} | {abs(trad_pen - aigcd_pen):8.4f}"
            )

            # reward rows at requested x values
            for x in xs:
                out_rew = _eval_pair(agg, np.array([x]), np.array([1.0]))[0]
                aigcd_rew = 100.0 * (out_rew / max(x, EPS) - 1.0)
                if mode == "AG":
                    trad_rew = float(analytic_ag_reward_rel(x, np.array([w]))[0])
                else:
                    trad_rew = float(analytic_ah_reward_rel(x, np.array([w]))[0])
                print(
                    f"{mode:<4} | {'reward':<7} | {x:>4.2f} | {w:>4.2f} | "
                    f"{trad_rew:11.4f} | {aigcd_rew:9.4f} | {abs(trad_rew - aigcd_rew):8.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate LSP CPA visualization using AIGCD aggregator")
    parser.add_argument("--show", action="store_true", help="Show matplotlib figure")
    parser.add_argument("--save", type=str, default="", help="Save figure path, e.g. lab/aggregator/lsp_cpa_generic_gl.png")
    args = parser.parse_args()

    W = np.linspace(0.05, 0.98, 300)
    xs = [0.25, 0.5, 0.75]
    w_samples = [0.25, 0.5, 0.75]

    _print_traditional_vs_aigcd_table(xs, w_samples)
    _print_transform_matrices(w_samples)

    ag_pen_gl = penalty_rel("AG", W)
    ah_pen_gl = penalty_rel("AH", W)
    ag_rewards_gl = [reward_rel("AG", x, W) for x in xs]
    ah_rewards_gl = [reward_rel("AH", x, W) for x in xs]

    ag_pen_analytic = analytic_ag_penalty_rel(W)
    ah_pen_analytic = analytic_ah_penalty_rel(W)
    ag_rewards_analytic = [analytic_ag_reward_rel(x, W) for x in xs]
    ah_rewards_analytic = [analytic_ah_reward_rel(x, W) for x in xs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for x, y in zip(xs, ag_rewards_analytic):
        ax.plot(W, y, linewidth=2, label=f"Reward x={x} (analytic)")
    ax.plot(W, ag_pen_analytic, linewidth=2, label="Penalty (analytic)")

    # Overlay AIGCD as dots to verify near-perfect match.
    dot_stride = 6
    for x, y in zip(xs, ag_rewards_gl):
        ax.plot(
            W[::dot_stride],
            y[::dot_stride],
            linestyle="None",
            marker="o",
            markersize=3,
            alpha=0.9,
            label=f"Reward x={x} (AIGCD)",
        )
    ax.plot(
        W[::dot_stride],
        ag_pen_gl[::dot_stride],
        linestyle="None",
        marker="o",
        markersize=3,
        alpha=0.9,
        label="Penalty (AIGCD)",
    )

    ax.set_title("AG CPA: analytic lines + AIGCD dots")
    ax.set_xlabel("Weight W1")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Penalty [-%] and reward [+%]")
    ax.set_xlim(0, 1)
    ax.set_ylim(-80, 100)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(-80, 110, 10))
    ax.grid(True)

    ax = axes[1]
    for x, y in zip(xs, ah_rewards_analytic):
        ax.plot(W, y, linewidth=2, label=f"Reward x={x} (analytic)")
    ax.plot(W, ah_pen_analytic, linewidth=2, label="Penalty (analytic)")

    for x, y in zip(xs, ah_rewards_gl):
        ax.plot(
            W[::dot_stride],
            y[::dot_stride],
            linestyle="None",
            marker="o",
            markersize=3,
            alpha=0.9,
            label=f"Reward x={x} (AIGCD)",
        )
    ax.plot(
        W[::dot_stride],
        ah_pen_gl[::dot_stride],
        linestyle="None",
        marker="o",
        markersize=3,
        alpha=0.9,
        label="Penalty (AIGCD)",
    )

    ax.set_title("AH CPA: analytic lines + AIGCD dots")
    ax.set_xlabel("Weight W1")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Penalty [-%] and reward [+%]")
    ax.set_xlim(0, 1)
    ax.set_ylim(-100, 60)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(-100, 70, 10))
    ax.grid(True)

    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)

    plt.tight_layout()

    # Weight-inferred interpretation (constant here because weight mode is static).
    ag_ref = _build_cpa_agg("AG", 0.5)
    ah_ref = _build_cpa_agg("AH", 0.5)
    ag_a, ag_pen_inf, ag_rew_inf = inferred_penalty_reward_from_weights(ag_ref)
    ah_a, ah_pen_inf, ah_rew_inf = inferred_penalty_reward_from_weights(ah_ref)

    print("\nWeight-inferred behavior (from anchor weights/andness):")
    print(f"AG: andness={ag_a:.3f}, penalty_propensity={ag_pen_inf:.1f}%, reward_propensity={ag_rew_inf:.1f}%")
    print(f"AH: andness={ah_a:.3f}, penalty_propensity={ah_pen_inf:.1f}%, reward_propensity={ah_rew_inf:.1f}%")

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {args.save}")

    if args.show or not args.save:
        plt.show()


if __name__ == "__main__":
    main()
