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
# 2. Target operator families
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

def make_target(X, Y, w, target_type):
    target_type = target_type.upper()
    if target_type == "H":
        return partial_absorption_h(X, Y, w)
    if target_type == "G":
        return partial_absorption_g(X, Y, w)
    raise ValueError(f"Unsupported target_type={target_type}. Use 'H' or 'G'.")


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
# 5. Continuous anchor family
# ============================================================

ANCHOR_NAMES = ["min", "harmonic", "geometric", "arithmetic", "quadratic", "max"]
PARAM_ANCHOR_NAMES = ["harmonic", "geometric", "arithmetic", "quadratic"]


def unpack_anchor_params(anchor_param_logits):
    """
    Convert unconstrained logits into continuous weights in (0,1)
    for H, G, A, Q.
    """
    # order: harmonic, geometric, arithmetic, quadratic
    return {
        "harmonic": sigmoid(anchor_param_logits[0]),
        "geometric": sigmoid(anchor_param_logits[1]),
        "arithmetic": sigmoid(anchor_param_logits[2]),
        "quadratic": sigmoid(anchor_param_logits[3]),
    }


def evaluate_anchor_family(X, Y, anchor_param_logits):
    """
    Returns:
        names: list of anchor names
        stack: [k, n, n]
        anchor_params: dict of learned continuous weights
    """
    p = unpack_anchor_params(anchor_param_logits)

    stack = np.stack([
        np.minimum(X, Y),                               # min
        weighted_harmonic(X, Y, p["harmonic"]),        # H_wH
        weighted_geometric(X, Y, p["geometric"]),      # G_wG
        weighted_arithmetic(X, Y, p["arithmetic"]),    # A_wA
        weighted_quadratic(X, Y, p["quadratic"]),      # Q_wQ
        np.maximum(X, Y),                               # max
    ], axis=0)

    return ANCHOR_NAMES, stack, p


# ============================================================
# 6. Final latent fusion helpers
# ============================================================

def fuse_latents(z1, z2, fusion_weights, fusion_param_logits):
    """
    fusion_weights: softmax over [H, A, G]
    fusion_param_logits: learnable continuous weights for H, A, G
    """
    wH = sigmoid(fusion_param_logits[0])
    wA = sigmoid(fusion_param_logits[1])
    wG = sigmoid(fusion_param_logits[2])

    fh = weighted_harmonic(z1, z2, wH)
    fa = weighted_arithmetic(z1, z2, wA)
    fg = weighted_geometric(z1, z2, wG)

    pred = fusion_weights[0] * fh + fusion_weights[1] * fa + fusion_weights[2] * fg
    fusion_params = {"H": wH, "A": wA, "G": wG}
    return pred, fusion_params


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
# 8. Models
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
        "name": "Model 1: convex anchor blend",
        "anchor_names": ANCHOR_NAMES,
        "alpha": alpha,
        "anchor_params": anchor_params,
        "pred": pred,
        "rmse": rmse(pred, target),
    }


def fit_model_2_affine_output(X, Y, target):
    k = len(ANCHOR_NAMES)

    def obj(theta):
        alpha_logits = theta[:k]
        anchor_param_logits = theta[k:k+4]
        a = theta[k+4]
        b = theta[k+5]

        alpha = softmax(alpha_logits)
        _, stack, _ = evaluate_anchor_family(X, Y, anchor_param_logits)
        base = np.tensordot(alpha, stack, axes=(0, 0))
        pred = a * base + b
        return np.mean((pred - target) ** 2)

    x0 = np.zeros(k + 6)
    x0[k+4] = 1.0
    x0[k+5] = 0.0

    res = minimize(obj, x0, method="L-BFGS-B")

    theta = res.x
    alpha = softmax(theta[:k])
    _, stack, anchor_params = evaluate_anchor_family(X, Y, theta[k:k+4])
    a = theta[k+4]
    b = theta[k+5]
    base = np.tensordot(alpha, stack, axes=(0, 0))
    pred = a * base + b

    return {
        "name": "Model 2: affine calibrated anchor blend",
        "anchor_names": ANCHOR_NAMES,
        "alpha": alpha,
        "anchor_params": anchor_params,
        "a": a,
        "b": b,
        "pred": pred,
        "rmse": rmse(pred, target),
    }


def fit_model_3_blender(X, Y, target):
    k = len(ANCHOR_NAMES)

    def obj(theta):
        p_logits = theta[:k]
        q_logits = theta[k:2*k]
        anchor_param_logits = theta[2*k:2*k+4]
        fusion_logits = theta[2*k+4:2*k+7]
        fusion_param_logits = theta[2*k+7:2*k+10]

        p = softmax(p_logits)
        q = softmax(q_logits)
        fusion_weights = softmax(fusion_logits)

        _, stack, _ = evaluate_anchor_family(X, Y, anchor_param_logits)
        z1 = np.tensordot(p, stack, axes=(0, 0))
        z2 = np.tensordot(q, stack, axes=(0, 0))

        pred, _ = fuse_latents(z1, z2, fusion_weights, fusion_param_logits)
        return np.mean((pred - target) ** 2)

    x0 = np.zeros(2*k + 10)
    res = minimize(obj, x0, method="L-BFGS-B")

    theta = res.x
    p = softmax(theta[:k])
    q = softmax(theta[k:2*k])
    anchor_param_logits = theta[2*k:2*k+4]
    fusion_weights = softmax(theta[2*k+4:2*k+7])
    fusion_param_logits = theta[2*k+7:2*k+10]

    _, stack, anchor_params = evaluate_anchor_family(X, Y, anchor_param_logits)
    z1 = np.tensordot(p, stack, axes=(0, 0))
    z2 = np.tensordot(q, stack, axes=(0, 0))
    pred, fusion_params = fuse_latents(z1, z2, fusion_weights, fusion_param_logits)

    return {
        "name": "Model 3: anchor blender + final H/A/G fusion",
        "anchor_names": ANCHOR_NAMES,
        "p": p,
        "q": q,
        "anchor_params": anchor_params,
        "fusion_weights": fusion_weights,
        "fusion_params": fusion_params,
        "u1": z1,
        "u2": z2,
        "pred": pred,
        "rmse": rmse(pred, target),
    }


def fit_model_4_input_transform(X, Y, target):
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
        return np.mean((pred - target) ** 2)

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
        "name": "Model 4: input linear transform + anchor blend",
        "anchor_names": ANCHOR_NAMES,
        "R": R,
        "alpha": alpha,
        "anchor_params": anchor_params,
        "u": u,
        "v": v,
        "pred": pred,
        "rmse": rmse(pred, target),
    }


def fit_model_5_full(X, Y, target):
    k = len(ANCHOR_NAMES)

    def obj(theta):
        R_logits = theta[:4].reshape(2, 2)
        p_logits = theta[4:4+k]
        q_logits = theta[4+k:4+2*k]
        anchor_param_logits = theta[4+2*k:4+2*k+4]
        fusion_logits = theta[4+2*k+4:4+2*k+7]
        fusion_param_logits = theta[4+2*k+7:4+2*k+10]

        R = row_softmax(R_logits)
        p = softmax(p_logits)
        q = softmax(q_logits)
        fusion_weights = softmax(fusion_logits)

        u = R[0, 0] * X + R[0, 1] * Y
        v = R[1, 0] * X + R[1, 1] * Y

        _, stack, _ = evaluate_anchor_family(u, v, anchor_param_logits)
        z1 = np.tensordot(p, stack, axes=(0, 0))
        z2 = np.tensordot(q, stack, axes=(0, 0))

        pred, _ = fuse_latents(z1, z2, fusion_weights, fusion_param_logits)
        return np.mean((pred - target) ** 2)

    x0 = np.zeros(4 + 2*k + 10)
    x0[:4] = np.array([3.0, 0.0, 0.0, 3.0])  # near identity

    res = minimize(obj, x0, method="L-BFGS-B")

    theta = res.x
    R = row_softmax(theta[:4].reshape(2, 2))
    p = softmax(theta[4:4+k])
    q = softmax(theta[4+k:4+2*k])
    anchor_param_logits = theta[4+2*k:4+2*k+4]
    fusion_weights = softmax(theta[4+2*k+4:4+2*k+7])
    fusion_param_logits = theta[4+2*k+7:4+2*k+10]

    u = R[0, 0] * X + R[0, 1] * Y
    v = R[1, 0] * X + R[1, 1] * Y

    _, stack, anchor_params = evaluate_anchor_family(u, v, anchor_param_logits)
    z1 = np.tensordot(p, stack, axes=(0, 0))
    z2 = np.tensordot(q, stack, axes=(0, 0))
    pred, fusion_params = fuse_latents(z1, z2, fusion_weights, fusion_param_logits)

    return {
        "name": "Model 5: input transform + blender + final H/A/G fusion",
        "anchor_names": ANCHOR_NAMES,
        "R": R,
        "p": p,
        "q": q,
        "anchor_params": anchor_params,
        "fusion_weights": fusion_weights,
        "fusion_params": fusion_params,
        "u": u,
        "v": v,
        "z1": z1,
        "z2": z2,
        "pred": pred,
        "rmse": rmse(pred, target),
    }


# ============================================================
# 9. Reporting
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

    if "a" in result and "b" in result:
        print(f"\nAffine calibration: a={result['a']:.6f}, b={result['b']:.6f}")

    if "p" in result:
        print("\nBlender p:")
        for n, a in zip(result["anchor_names"], result["p"]):
            print(f"  {n:>10s}: {a:.6f}")

    if "q" in result:
        print("\nBlender q:")
        for n, a in zip(result["anchor_names"], result["q"]):
            print(f"  {n:>10s}: {a:.6f}")

    if "fusion_weights" in result:
        fw = result["fusion_weights"]
        print("\nFinal fusion weights [H, A, G]:")
        print(f"  H: {fw[0]:.6f}")
        print(f"  A: {fw[1]:.6f}")
        print(f"  G: {fw[2]:.6f}")

    if "fusion_params" in result:
        print("\nFinal fusion operator parameters:")
        for k, v in result["fusion_params"].items():
            print(f"  {k:>10s}: {v:.6f}")

    if "R" in result:
        print("\nInput transform R:")
        print(result["R"])

    print("=" * 80)


# ============================================================
# 10. Main
# ============================================================

def run_all(w_target=0.1, target_type="H", n=121, make_plots=True):
    xs, ys, X, Y = make_grid(n=n)
    target = make_target(X, Y, w_target, target_type)

    print(f"\nTarget type = {target_type.upper()}, w = {w_target}")
    if target_type.upper() == "H":
        print("  P_H(x,y;w) = H_w(x, A_w(x,y))")
    else:
        print("  P_G(x,y;w) = G_w(x, A_w(x,y))")
    print()

    models = [
        fit_model_1_convex_anchors(X, Y, target),
        fit_model_2_affine_output(X, Y, target),
        fit_model_3_blender(X, Y, target),
        fit_model_4_input_transform(X, Y, target),
        fit_model_5_full(X, Y, target),
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Partial absorption with continuous learnable anchor parameters"
    )
    parser.add_argument(
        "--w",
        type=float,
        default=0.1,
        help="Target weight w in P_H(x,y)=H_w(x,A_w(x,y)) or P_G(x,y)=G_w(x,A_w(x,y))"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="H",
        choices=["H", "G", "h", "g"],
        help="Target family: H for harmonic absorption, G for geometric absorption"
    )
    parser.add_argument("--n", type=int, default=121, help="Grid size")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        w_target=args.w,
        target_type=args.target,
        n=args.n,
        make_plots=not args.no_plots,
    )