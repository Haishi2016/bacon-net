import argparse
import math
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

def partial_absorption(x, y, w, eps=EPS):
    # P(x,y;w) = H_w(x, A_w(x,y))
    a = weighted_arithmetic(x, y, w)
    return weighted_harmonic(x, a, w, eps=eps)


# ============================================================
# 2. Anchor bank construction
# ============================================================

def make_anchor_bank(weight_params=None):
    """
    Returns a dict name -> callable(x,y).
    Weighted anchors are created for each lambda in weight_params.
    """
    if weight_params is None:
        weight_params = [0.1, 0.25, 0.5, 0.75, 0.9]

    anchors = {}

    # Unweighted extrema
    anchors["min"] = lambda x, y: np.minimum(x, y)
    anchors["max"] = lambda x, y: np.maximum(x, y)

    # Weighted mean family
    for w in weight_params:
        ws = f"{w:.2f}"

        anchors[f"H_{ws}"] = lambda x, y, ww=w: weighted_harmonic(x, y, ww)
        anchors[f"G_{ws}"] = lambda x, y, ww=w: weighted_geometric(x, y, ww)
        anchors[f"A_{ws}"] = lambda x, y, ww=w: weighted_arithmetic(x, y, ww)
        anchors[f"Q_{ws}"] = lambda x, y, ww=w: weighted_quadratic(x, y, ww)

    return anchors


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

def row_softmax(mat):
    out = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        out[i] = softmax(mat[i])
    return out

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def anchor_stack(X, Y, anchors_dict):
    names = list(anchors_dict.keys())
    stack = np.stack([anchors_dict[name](X, Y) for name in names], axis=0)  # [k,n,n]
    return names, stack


# ============================================================
# 5. Plotting
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
    plot_surface(axes[0], X, Y, target, "Target P(x,y)")
    plot_surface(axes[1], X, Y, pred, result["name"])
    plot_surface(axes[2], X, Y, err, "Error = pred - target")
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Models
# ============================================================

def fit_model_1_convex_anchors(X, Y, target, anchors_dict):
    names, stack = anchor_stack(X, Y, anchors_dict)
    k = stack.shape[0]
    A = stack.reshape(k, -1).T
    b = target.reshape(-1)

    def obj(theta):
        alpha = softmax(theta)
        pred = A @ alpha
        return np.mean((pred - b) ** 2)

    x0 = np.zeros(k)
    res = minimize(obj, x0, method="L-BFGS-B")
    alpha = softmax(res.x)
    pred = np.tensordot(alpha, stack, axes=(0, 0))

    return {
        "name": "Model 1: convex anchor blend",
        "anchor_names": names,
        "alpha": alpha,
        "pred": pred,
        "rmse": rmse(pred, target),
    }

def fit_model_2_affine_output(X, Y, target, anchors_dict):
    names, stack = anchor_stack(X, Y, anchors_dict)
    k = stack.shape[0]
    A = stack.reshape(k, -1).T
    b_target = target.reshape(-1)

    def obj(theta):
        alpha_logits = theta[:k]
        a = theta[k]
        b = theta[k + 1]
        alpha = softmax(alpha_logits)
        base = A @ alpha
        pred = a * base + b
        return np.mean((pred - b_target) ** 2)

    x0 = np.zeros(k + 2)
    x0[k] = 1.0
    x0[k + 1] = 0.0

    res = minimize(obj, x0, method="L-BFGS-B")
    theta = res.x
    alpha = softmax(theta[:k])
    a = theta[k]
    b = theta[k + 1]
    base = np.tensordot(alpha, stack, axes=(0, 0))
    pred = a * base + b

    return {
        "name": "Model 2: affine calibrated anchor blend",
        "anchor_names": names,
        "alpha": alpha,
        "a": a,
        "b": b,
        "pred": pred,
        "rmse": rmse(pred, target),
    }

def fit_model_3_blender(X, Y, target, anchors_dict):
    names, stack = anchor_stack(X, Y, anchors_dict)
    k = stack.shape[0]

    def obj(theta):
        p_logits = theta[:k]
        q_logits = theta[k:2*k]
        beta_logit = theta[2*k]

        p = softmax(p_logits)
        q = softmax(q_logits)
        beta = sigmoid(beta_logit)

        u1 = np.tensordot(p, stack, axes=(0, 0))
        u2 = np.tensordot(q, stack, axes=(0, 0))

        # final fusion uses standard H/A on latent channels
        pred_h = weighted_harmonic(u1, u2, 0.5)
        pred_a = weighted_arithmetic(u1, u2, 0.5)
        pred = beta * pred_h + (1.0 - beta) * pred_a

        return np.mean((pred - target) ** 2)

    x0 = np.zeros(2 * k + 1)
    res = minimize(obj, x0, method="L-BFGS-B")

    theta = res.x
    p = softmax(theta[:k])
    q = softmax(theta[k:2*k])
    beta = sigmoid(theta[2*k])

    u1 = np.tensordot(p, stack, axes=(0, 0))
    u2 = np.tensordot(q, stack, axes=(0, 0))
    pred_h = weighted_harmonic(u1, u2, 0.5)
    pred_a = weighted_arithmetic(u1, u2, 0.5)
    pred = beta * pred_h + (1.0 - beta) * pred_a

    return {
        "name": "Model 3: anchor blender + final H/A fusion",
        "anchor_names": names,
        "p": p,
        "q": q,
        "beta": beta,
        "u1": u1,
        "u2": u2,
        "pred": pred,
        "rmse": rmse(pred, target),
    }

def fit_model_4_input_transform(X, Y, target, anchors_dict):
    names = list(anchors_dict.keys())
    k = len(names)

    def obj(theta):
        R_logits = theta[:4].reshape(2, 2)
        alpha_logits = theta[4:4 + k]

        R = row_softmax(R_logits)
        alpha = softmax(alpha_logits)

        u = R[0, 0] * X + R[0, 1] * Y
        v = R[1, 0] * X + R[1, 1] * Y

        phi = np.stack([anchors_dict[name](u, v) for name in names], axis=0)
        pred = np.tensordot(alpha, phi, axes=(0, 0))
        return np.mean((pred - target) ** 2)

    x0 = np.zeros(4 + k)
    x0[:4] = np.array([3.0, 0.0, 0.0, 3.0])  # near identity

    res = minimize(obj, x0, method="L-BFGS-B")
    theta = res.x

    R = row_softmax(theta[:4].reshape(2, 2))
    alpha = softmax(theta[4:4 + k])

    u = R[0, 0] * X + R[0, 1] * Y
    v = R[1, 0] * X + R[1, 1] * Y
    phi = np.stack([anchors_dict[name](u, v) for name in names], axis=0)
    pred = np.tensordot(alpha, phi, axes=(0, 0))

    return {
        "name": "Model 4: input linear transform + anchor blend",
        "anchor_names": names,
        "R": R,
        "alpha": alpha,
        "u": u,
        "v": v,
        "pred": pred,
        "rmse": rmse(pred, target),
    }

def fit_model_5_full(X, Y, target, anchors_dict):
    names = list(anchors_dict.keys())
    k = len(names)

    def obj(theta):
        R_logits = theta[:4].reshape(2, 2)
        p_logits = theta[4:4 + k]
        q_logits = theta[4 + k:4 + 2*k]
        beta_logit = theta[4 + 2*k]

        R = row_softmax(R_logits)
        p = softmax(p_logits)
        q = softmax(q_logits)
        beta = sigmoid(beta_logit)

        u = R[0, 0] * X + R[0, 1] * Y
        v = R[1, 0] * X + R[1, 1] * Y

        phi = np.stack([anchors_dict[name](u, v) for name in names], axis=0)

        z1 = np.tensordot(p, phi, axes=(0, 0))
        z2 = np.tensordot(q, phi, axes=(0, 0))

        pred_h = weighted_harmonic(z1, z2, 0.5)
        pred_a = weighted_arithmetic(z1, z2, 0.5)
        pred = beta * pred_h + (1.0 - beta) * pred_a

        return np.mean((pred - target) ** 2)

    x0 = np.zeros(4 + 2*k + 1)
    x0[:4] = np.array([3.0, 0.0, 0.0, 3.0])  # near identity

    res = minimize(obj, x0, method="L-BFGS-B")
    theta = res.x

    R = row_softmax(theta[:4].reshape(2, 2))
    p = softmax(theta[4:4 + k])
    q = softmax(theta[4 + k:4 + 2*k])
    beta = sigmoid(theta[4 + 2*k])

    u = R[0, 0] * X + R[0, 1] * Y
    v = R[1, 0] * X + R[1, 1] * Y

    phi = np.stack([anchors_dict[name](u, v) for name in names], axis=0)
    z1 = np.tensordot(p, phi, axes=(0, 0))
    z2 = np.tensordot(q, phi, axes=(0, 0))

    pred_h = weighted_harmonic(z1, z2, 0.5)
    pred_a = weighted_arithmetic(z1, z2, 0.5)
    pred = beta * pred_h + (1.0 - beta) * pred_a

    return {
        "name": "Model 5: input transform + blender + final H/A fusion",
        "anchor_names": names,
        "R": R,
        "p": p,
        "q": q,
        "beta": beta,
        "u": u,
        "v": v,
        "z1": z1,
        "z2": z2,
        "pred": pred,
        "rmse": rmse(pred, target),
    }


# ============================================================
# 7. Reporting
# ============================================================

def print_result_summary(result):
    print("=" * 72)
    print(result["name"])
    print(f"RMSE: {result['rmse']:.8f}")

    if "alpha" in result:
        print("\nAnchor weights:")
        for n, a in zip(result["anchor_names"], result["alpha"]):
            print(f"  {n:>10s}: {a:.6f}")

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

    if "beta" in result:
        print(f"\nFinal fusion beta (H weight): {result['beta']:.6f}")

    if "R" in result:
        print("\nInput transform R:")
        print(result["R"])

    print("=" * 72)


# ============================================================
# 8. Main
# ============================================================

def run_all(w_target=0.1, n=121, weight_params=None, make_plots=True):
    xs, ys, X, Y = make_grid(n=n)
    target = partial_absorption(X, Y, w_target)

    if weight_params is None:
        # include target weight explicitly to help the bank match the target family
        base = [0.1, 0.25, 0.5, 0.75, 0.9]
        if all(abs(w_target - b) > 1e-12 for b in base):
            base.append(w_target)
        weight_params = sorted(set(base))

    anchors = make_anchor_bank(weight_params=weight_params)

    print(f"\nTarget: P(x,y) = H_w(x, A_w(x,y)), with w = {w_target}")
    print(f"Weighted anchor parameters = {weight_params}\n")

    models = [
        fit_model_1_convex_anchors(X, Y, target, anchors),
        fit_model_2_affine_output(X, Y, target, anchors),
        fit_model_3_blender(X, Y, target, anchors),
        fit_model_4_input_transform(X, Y, target, anchors),
        fit_model_5_full(X, Y, target, anchors),
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
    parser = argparse.ArgumentParser(description="Partial absorption with weighted anchor bank")
    parser.add_argument("--w", type=float, default=0.1, help="Target weight w in P(x,y)=H_w(x,A_w(x,y))")
    parser.add_argument("--n", type=int, default=121, help="Grid size")
    parser.add_argument(
        "--anchor-weights",
        type=str,
        default="0.1,0.25,0.5,0.75,0.9",
        help="Comma-separated weighted-anchor parameters"
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    weight_params = [float(x.strip()) for x in args.anchor_weights.split(",") if x.strip()]
    run_all(
        w_target=args.w,
        n=args.n,
        weight_params=weight_params,
        make_plots=not args.no_plots,
    )