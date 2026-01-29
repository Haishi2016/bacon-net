import numpy as np
import matplotlib.pyplot as plt

def softmax(z, axis=0):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

# --- Grid on the 2D hypercube [0,1]^2 ---
n = 151
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

Z_min = np.minimum(X, Y)
Z_avg = 0.5 * (X + Y)
Z_max = np.maximum(X, Y)

# --- Softmax interpolation over [min, avg, max] ---
# You can choose either:
# (A) fixed mixture weights (constant over the square), or
# (B) input-dependent weights (vary over the square)

use_input_dependent = True

if not use_input_dependent:
    # (A) Fixed weights example: mostly avg, some min/max
    w = np.array([0.2, 0.6, 0.2])  # [min, avg, max]
    Z_soft = w[0]*Z_min + w[1]*Z_avg + w[2]*Z_max
else:
    # (B) Input-dependent weights via softmax over "scores"
    # Intuition:
    # - When X and Y are close: favor AND-ish (min)
    # - When one is much larger: favor OR-ish (max)
    # - Otherwise: favor compromise (avg)
    #
    # You can tune these. Bigger "tau" => sharper switching.
    tau = 10.0

    closeness = -np.abs(X - Y)             # highest when X≈Y
    imbalance =  np.abs(X - Y)             # highest when one dominates
    midness   = -np.abs((X + Y) - 1.0)     # favors around X+Y≈1 (optional)

    # Scores for [min, avg, max]
    # Feel free to tweak coefficients to change behavior.
    s_min =  1.2*closeness + 0.2*(X+Y) - 0.2*imbalance
    s_avg =  0.4*midness   + 0.2*(X+Y) - 0.1*imbalance
    s_max =  1.2*imbalance + 0.2*(X+Y) - 0.2*closeness

    S = tau * np.stack([s_min, s_avg, s_max], axis=0)  # shape (3, n, n)
    W = softmax(S, axis=0)  # shape (3, n, n), sums to 1 at each (x,y)

    Z_soft = W[0]*Z_min + W[1]*Z_avg + W[2]*Z_max

# --- Plotting ---
# 3D surface plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Light alpha so you can see overlap
ax.plot_surface(X, Y, Z_min, rstride=3, cstride=3, alpha=0.35)
ax.plot_surface(X, Y, Z_avg, rstride=3, cstride=3, alpha=0.35)
ax.plot_surface(X, Y, Z_max, rstride=3, cstride=3, alpha=0.35)
ax.plot_surface(X, Y, Z_soft, rstride=3, cstride=3, alpha=0.65)

ax.set_title("Hypercube [0,1]^2: min, avg, max, and softmax interpolation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("aggregation")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Legend hack for 3D surfaces
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(label="min(x,y)"),
    mpatches.Patch(label="avg(x,y)"),
    mpatches.Patch(label="max(x,y)"),
    mpatches.Patch(label="softmax-mix"),
]
ax.legend(handles=legend_patches, loc="upper left")

plt.tight_layout()
plt.show()

# 2D contour comparison (easier to interpret)
fig2, axs = plt.subplots(2, 2, figsize=(12, 10))
levels = np.linspace(0, 1, 11)

cs0 = axs[0,0].contourf(X, Y, Z_min, levels=levels)
axs[0,0].set_title("min(x,y)")
plt.colorbar(cs0, ax=axs[0,0])

cs1 = axs[0,1].contourf(X, Y, Z_avg, levels=levels)
axs[0,1].set_title("avg(x,y)")
plt.colorbar(cs1, ax=axs[0,1])

cs2 = axs[1,0].contourf(X, Y, Z_max, levels=levels)
axs[1,0].set_title("max(x,y)")
plt.colorbar(cs2, ax=axs[1,0])

cs3 = axs[1,1].contourf(X, Y, Z_soft, levels=levels)
axs[1,1].set_title("softmax interpolation")
plt.colorbar(cs3, ax=axs[1,1])

for a in axs.ravel():
    a.set_xlabel("x")
    a.set_ylabel("y")
    a.set_aspect("equal", "box")

plt.tight_layout()
plt.show()

# Optional: visualize the softmax weights themselves (only if input-dependent)
if use_input_dependent:
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["weight for min", "weight for avg", "weight for max"]
    for i in range(3):
        cs = axs3[i].contourf(X, Y, W[i], levels=np.linspace(0,1,11))
        axs3[i].set_title(titles[i])
        axs3[i].set_xlabel("x")
        axs3[i].set_ylabel("y")
        axs3[i].set_aspect("equal", "box")
        plt.colorbar(cs, ax=axs3[i])
    plt.tight_layout()
    plt.show()
