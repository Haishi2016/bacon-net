import numpy as np
import matplotlib.pyplot as plt

def softmax(z, axis=0):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

# ---- Grid on the 2D hypercube [0,1]^2 ----
n = 201
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

# Base operators
Z_min = np.minimum(X, Y)
Z_avg = 0.5 * (X + Y)
Z_max = np.maximum(X, Y)

# ---- Softmax interpolation over [min, avg, max] ----
# W(x,y) = softmax([s_min, s_avg, s_max]) at each point
tau = 12.0  # bigger => sharper regime selection

# Scores: tweak these to get different gating behavior
closeness = -np.abs(X - Y)          # best (0) when X≈Y
imbalance =  np.abs(X - Y)          # high when one dominates
both_high =  (X + Y)                # in [0,2]
midness   = -np.abs((X + Y) - 1.0)  # best (0) near X+Y≈1

# Scores for [min, avg, max]
s_min =  1.6*closeness + 0.25*both_high - 0.25*imbalance
s_avg =  1.0*midness   + 0.15*both_high - 0.10*imbalance
s_max =  1.6*imbalance + 0.25*both_high - 0.25*(-closeness)

S = tau * np.stack([s_min, s_avg, s_max], axis=0)  # (3, n, n)
W = softmax(S, axis=0)                              # (3, n, n)

Z_soft = W[0]*Z_min + W[1]*Z_avg + W[2]*Z_max

# ---- Diagnostics ----
print("W shape:", W.shape)  # (3, n, n)
print("Weight ranges:")
print("  min-weight:", float(W[0].min()), float(W[0].max()))
print("  avg-weight:", float(W[1].min()), float(W[1].max()))
print("  max-weight:", float(W[2].min()), float(W[2].max()))

sumW = W[0] + W[1] + W[2]
print("Sum(W) range (should be ~1):", float(sumW.min()), float(sumW.max()))
print("Max |Sum(W)-1|:", float(np.max(np.abs(sumW - 1.0))))

# ---- Plot 1: 2D contour comparison ----
levels = np.linspace(0, 1, 21)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

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
axs[1,1].set_title("softmax mix")
plt.colorbar(cs3, ax=axs[1,1])

for a in axs.ravel():
    a.set_xlabel("x")
    a.set_ylabel("y")
    a.set_aspect("equal", "box")

plt.tight_layout()
plt.show()

# ---- Plot 2: weights (the "matrix" over the hypercube) ----
fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4.5))
titles = ["W_min (weight on min)", "W_avg (weight on avg)", "W_max (weight on max)"]
for i in range(3):
    cs = axs2[i].contourf(X, Y, W[i], levels=np.linspace(0, 1, 21))
    axs2[i].set_title(titles[i])
    axs2[i].set_xlabel("x")
    axs2[i].set_ylabel("y")
    axs2[i].set_aspect("equal", "box")
    plt.colorbar(cs, ax=axs2[i])
plt.tight_layout()
plt.show()

# ---- Plot 3: difference map (proves it's not just avg) ----
Z_diff = Z_soft - Z_avg
plt.figure(figsize=(7,6))
cs = plt.contourf(X, Y, Z_diff, levels=21)
plt.title("Z_soft - Z_avg (where the softmax mix deviates from avg)")
plt.xlabel("x"); plt.ylabel("y")
plt.gca().set_aspect("equal", "box")
plt.colorbar(cs)
plt.tight_layout()
plt.show()

# ---- Plot 4: 3D surfaces (softmix + faint avg reference) ----
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig3 = plt.figure(figsize=(11, 8))
ax = fig3.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z_soft, rstride=4, cstride=4, alpha=0.95)
ax.plot_surface(X, Y, Z_avg,  rstride=8, cstride=8, alpha=0.15)

ax.set_title("Softmax-mix surface (with faint avg reference)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("aggregation")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

plt.tight_layout()
plt.show()

# ---- Point check ----
def nearest_idx(val, grid):
    return int(np.argmin(np.abs(grid - val)))

x0, y0 = 0.9, 0.4
ix = nearest_idx(x0, x)
iy = nearest_idx(y0, y)

print("\nExample point check:")
print("  (x,y) =", float(x[ix]), float(y[iy]))
print("  min/avg/max =", float(Z_min[iy,ix]), float(Z_avg[iy,ix]), float(Z_max[iy,ix]))
print("  weights [min,avg,max] =", [float(W[0,iy,ix]), float(W[1,iy,ix]), float(W[2,iy,ix])])
print("  Z_soft =", float(Z_soft[iy,ix]))
