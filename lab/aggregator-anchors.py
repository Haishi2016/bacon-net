import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

EPS = 1e-9


# ============================================================
# 1. Base weighted operators
# ============================================================

def weighted_arithmetic(x, y, w):
    return w * x + (1.0 - w) * y

def weighted_harmonic(x, y, w, eps=EPS):
    x = np.maximum(x, eps)
    y = np.maximum(y, eps)
    return 1.0 / (w / x + (1.0 - w) / y)

def weighted_geometric(x, y, w, eps=EPS):
    x = np.maximum(x, eps)
    y = np.maximum(y, eps)
    return np.exp(w * np.log(x) + (1.0 - w) * np.log(y))

def weighted_quadratic(x, y, w):
    return np.sqrt(w * x**2 + (1.0 - w) * y**2)


# ============================================================
# 2. Target families
# ============================================================

def partial_absorption_h(x, y, w, eps=EPS):
    """
    P_H(x,y;w) = H_w(x, A_w(x,y))
    """
    a = weighted_arithmetic(x, y, w)
    return weighted_harmonic(x, a, w, eps=eps)

def partial_absorption_g(x, y, w, eps=EPS):
    """
    P_G(x,y;w) = G_w(x, A_w(x,y))
    """
    a = weighted_arithmetic(x, y, w)
    return weighted_geometric(x, a, w, eps=eps)

def make_target(X, Y, target_type, w=0.5):
    """
    Supported target types:
      A  -> weighted arithmetic A_w(x,y)
      H  -> weighted harmonic H_w(x,y)
      G  -> weighted geometric G_w(x,y)
      Q  -> weighted quadratic Q_w(x,y)
      PH -> partial absorption harmonic H_w(x, A_w(x,y))
      PG -> partial absorption geometric G_w(x, A_w(x,y))
    """
    t = target_type.upper()
    if t == "A":
        return weighted_arithmetic(X, Y, w)
    if t == "H":
        return weighted_harmonic(X, Y, w)
    if t == "G":
        return weighted_geometric(X, Y, w)
    if t == "Q":
        return weighted_quadratic(X, Y, w)
    if t == "PH":
        return partial_absorption_h(X, Y, w)
    if t == "PG":
        return partial_absorption_g(X, Y, w)
    raise ValueError(f"Unsupported target_type={target_type}")


# ============================================================
# 3. Grid
# ============================================================

def make_grid(n=101, eps=1e-3):
    xs = np.linspace(eps, 1.0, n)
    ys = np.linspace(eps, 1.0, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return xs, ys, X, Y


# ============================================================
# 4. Helpers
# ============================================================

def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def row_softmax(mat):
    out = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        out[i] = softmax(mat[i])
    return out

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


# ============================================================
# 5. Continuous learnable anchor family
# ============================================================

ANCHOR_NAMES = ["min", "harmonic", "geometric", "arithmetic", "quadratic", "max"]

def unpack_anchor_params(anchor_param_logits):
    """
    Continuous weights in (0,1) for H, G, A, Q.
    """
    return {
        "harmonic": sigmoid(anchor_param_logits[0]),
        "geometric": sigmoid(anchor_param_logits[1]),
        "arithmetic": sigmoid(anchor_param_logits[2]),
        "quadratic": sigmoid(anchor_param_logits[3]),
    }

def evaluate_anchor_family(X, Y, anchor_param_logits):
    """
    Returns:
      names
      stack: shape [k, n, n]
      anchor_params
    """
    p = unpack_anchor_params(anchor_param_logits)

    stack = np.stack([
        np.minimum(X, Y),
        weighted_harmonic(X, Y, p["harmonic"]),
        weighted_geometric(X, Y, p["geometric"]),
        weighted_arithmetic(X, Y, p["arithmetic"]),
        weighted_quadratic(X, Y, p["quadratic"]),
        np.maximum(X, Y),
    ], axis=0)

    return ANCHOR_NAMES, stack, p


# ============================================================
# 6. Identity regularization for R
# ============================================================

def identity_penalty(R):
    I = np.eye(2)
    return np.sum((R - I) ** 2)


# ============================================================
# 7. Plotting
# ============================================================

def plot_surface(ax, X, Y, Z, title):
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_slices(xs, ys, target, pred, title, y_values=(0.2, 0.5, 0.8)):
    fig, ax = plt.subplots(figsize=(9, 6))
    for y0 in y_values:
        idx = np.argmin(np.abs(ys - y0))
        ax.plot(xs, target[idx, :], label=f"Target, y={ys[idx]:.2f}")
        ax.plot(xs, pred[idx, :], "--", label=f"Pred, y={ys[idx]:.2f}")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_comparison(X, Y, target, result):
    pred = result["pred"]
    err = pred - target

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_surface(axes[0], X, Y, target, "Target")
    plot_surface(axes[1], X, Y, pred, result["name"])
    plot_surface(axes[2], X, Y, err, "Error = pred - target")
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Model 1: direct anchor blend
# ============================================================

def fit_model_1_convex_anchors(X, Y, target):
    k = len(ANCHOR_NAMES)

    def obj(theta):
        alpha_logits = theta[:k]
        anchor_param_logits = theta[k:k+4]

        alpha = softmax(alpha_logits)
        _, stack, _ = evaluate_anchor_family(X, Y, anchor_param_logits)
        pred = np.tensordot(alpha, stack, axes=(0, 0))
        return np.mean((pred - target) ** 2)

    x0 = np.zeros(k + 4)
    res = minimize(obj, x0, method="L-BFGS-B")

    theta = res.x
    alpha = softmax(theta[:k])
    _, stack, anchor_params = evaluate_anchor_family(X, Y, theta[k:k+4])
    pred = np.tensordot(alpha, stack, axes=(0, 0))

    return {
        "name": "Model 1: direct anchor blend",
        "anchor_names": ANCHOR_NAMES,
        "alpha": alpha,
        "anchor_params": anchor_params,
        "pred": pred,
        "rmse": rmse(pred, target),
    }


# ============================================================
# 9. Model 4: input transform + anchor blend
# ============================================================

def fit_model_4_input_transform(X, Y, target, lambda_R=0.0):
    k = len(ANCHOR_NAMES)

    def obj(theta):
        R_logits = theta[:4].reshape(2, 2)
        alpha_logits = theta[4:4+k]
        anchor_param_logits = theta[4+k:4+k+4]

        R = row_softmax(R_logits)
        alpha = softmax(alpha_logits)

        u = R[0, 0] * X + R[0, 1] * Y
        v = R[1, 0] * X + R[1, 1] * Y

        _, stack, _ = evaluate_anchor_family(u, v, anchor_param_logits)
        pred = np.tensordot(alpha, stack, axes=(0, 0))

        fit_loss = np.mean((pred - target) ** 2)
        reg_loss = lambda_R * identity_penalty(R)
        return fit_loss + reg_loss

    x0 = np.zeros(4 + k + 4)
    x0[:4] = np.array([3.0, 0.0, 0.0, 3.0])  # near identity

    res = minimize(obj, x0, method="L-BFGS-B")

    theta = res.x
    R = row_softmax(theta[:4].reshape(2, 2))
    alpha = softmax(theta[4:4+k])
    anchor_param_logits = theta[4+k:4+k+4]

    u = R[0, 0] * X + R[0, 1] * Y
    v = R[1, 0] * X + R[1, 1] * Y

    _, stack, anchor_params = evaluate_anchor_family(u, v, anchor_param_logits)
    pred = np.tensordot(alpha, stack, axes=(0, 0))

    return {
        "name": "Model 4: input transform + anchor blend",
        "anchor_names": ANCHOR_NAMES,
        "R": R,
        "alpha": alpha,
        "anchor_params": anchor_params,
        "u": u,
        "v": v,
        "pred": pred,
        "rmse": rmse(pred, target),
        "identity_penalty": identity_penalty(R),
    }


# ============================================================
# 10. Reporting
# ============================================================

def print_result_summary(result):
    print("=" * 80)
    print(result["name"])
    print(f"RMSE: {result['rmse']:.8f}")

    if "alpha" in result:
        print("\nAnchor weights:")
        for n, a in zip(result["anchor_names"], result["alpha"]):
            print(f"  {n:>10s}: {a:.6f}")

    if "anchor_params" in result:
        print("\nLearned anchor parameters:")
        for k, v in result["anchor_params"].items():
            print(f"  {k:>10s}: {v:.6f}")

    if "R" in result:
        print("\nInput transform R:")
        print(result["R"])
        print(f"Identity penalty ||R-I||^2: {result['identity_penalty']:.8f}")

    print("=" * 80)


# ============================================================
# 11. Main experiment runner
# ============================================================

def run_all(target_type="PH", w_target=0.5, n=121, lambda_R=0.0, make_plots=True):
    xs, ys, X, Y = make_grid(n=n)
    target = make_target(X, Y, target_type=target_type, w=w_target)

    print(f"\nTarget type = {target_type.upper()}, w = {w_target}")
    if target_type.upper() == "A":
        print("  target = A_w(x,y)")
    elif target_type.upper() == "H":
        print("  target = H_w(x,y)")
    elif target_type.upper() == "G":
        print("  target = G_w(x,y)")
    elif target_type.upper() == "Q":
        print("  target = Q_w(x,y)")
    elif target_type.upper() == "PH":
        print("  target = H_w(x, A_w(x,y))")
    elif target_type.upper() == "PG":
        print("  target = G_w(x, A_w(x,y))")
    print(f"Identity regularization lambda_R = {lambda_R}\n")

    models = [
        fit_model_1_convex_anchors(X, Y, target),
        fit_model_4_input_transform(X, Y, target, lambda_R=lambda_R),
    ]

    for m in models:
        print_result_summary(m)

    models_sorted = sorted(models, key=lambda z: z["rmse"])
    print("\nRanking by RMSE:")
    for i, m in enumerate(models_sorted, 1):
        print(f"{i}. {m['name']}  -> RMSE={m['rmse']:.8f}")

    best = models_sorted[0]

    if make_plots:
        plot_comparison(X, Y, target, best)
        plot_slices(xs, ys, target, best["pred"], f"Best model slices: {best['name']}")

    return models


# ============================================================
# 12. Sweep utility
# ============================================================

def run_sweep(target_type="A", w_values=None, n=121, lambda_R=0.0):
    if w_values is None:
        w_values = [0.1, 0.25, 0.5, 0.75, 0.9]

    rows = []
    for w in w_values:
        xs, ys, X, Y = make_grid(n=n)
        target = make_target(X, Y, target_type=target_type, w=w)

        m1 = fit_model_1_convex_anchors(X, Y, target)
        m4 = fit_model_4_input_transform(X, Y, target, lambda_R=lambda_R)

        rows.append({
            "w": w,
            "m1_rmse": m1["rmse"],
            "m4_rmse": m4["rmse"],
            "R11": m4["R"][0, 0],
            "R12": m4["R"][0, 1],
            "R21": m4["R"][1, 0],
            "R22": m4["R"][1, 1],
            "alpha_min": m4["alpha"][0],
            "alpha_H": m4["alpha"][1],
            "alpha_G": m4["alpha"][2],
            "alpha_A": m4["alpha"][3],
            "alpha_Q": m4["alpha"][4],
            "alpha_max": m4["alpha"][5],
            "wH": m4["anchor_params"]["harmonic"],
            "wG": m4["anchor_params"]["geometric"],
            "wA": m4["anchor_params"]["arithmetic"],
            "wQ": m4["anchor_params"]["quadratic"],
            "identity_penalty": m4["identity_penalty"],
        })

    print("\nSweep results:")
    print(
        "   w      m1_rmse    m4_rmse      R11      R12      R21      R22    "
        "alpha_A    wA     id_pen"
    )
    for r in rows:
        print(
            f"{r['w']:>5.2f}  "
            f"{r['m1_rmse']:>10.6f}  "
            f"{r['m4_rmse']:>10.6f}  "
            f"{r['R11']:>7.4f}  {r['R12']:>7.4f}  {r['R21']:>7.4f}  {r['R22']:>7.4f}  "
            f"{r['alpha_A']:>7.4f}  "
            f"{r['wA']:>6.4f}  "
            f"{r['identity_penalty']:>8.6f}"
        )

    return rows


# ============================================================
# 13. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="GL anchor learning with direct blend vs coordinate-transformed blend"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="PH",
        choices=["A", "H", "G", "Q", "PH", "PG", "a", "h", "g", "q", "ph", "pg"],
        help="Target family"
    )
    parser.add_argument(
        "--w",
        type=float,
        default=0.5,
        help="Weight parameter for the target"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=121,
        help="Grid size"
    )
    parser.add_argument(
        "--lambda-R",
        type=float,
        default=0.0,
        help="Identity regularization strength for Model 4"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep over preset w values instead of a single experiment"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plots"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sweep:
        run_sweep(
            target_type=args.target,
            w_values=[0.1, 0.25, 0.5, 0.75, 0.9],
            n=args.n,
            lambda_R=args.lambda_R,
        )
    else:
        run_all(
            target_type=args.target,
            w_target=args.w,
            n=args.n,
            lambda_R=args.lambda_R,
            make_plots=not args.no_plots,
        )