import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def softmax(z, axis=0):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

# ---- Grid on [0,1]^2 ----
n = 121  # animation: keep modest for speed
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

# Base operators
Z_min = np.minimum(X, Y)
Z_avg = 0.5 * (X + Y)
Z_max = np.maximum(X, Y)

# ---- Scores for [min, avg, max] ----
closeness = -np.abs(X - Y)          # best (0) when X≈Y
imbalance =  np.abs(X - Y)          # high when far apart
both_high = (X + Y)                 # in [0,2]
midness   = -np.abs((X + Y) - 1.0)  # best (0) near X+Y≈1

s_min =  1.6*closeness + 0.25*both_high - 0.25*imbalance
s_avg =  1.0*midness   + 0.15*both_high - 0.10*imbalance
s_max =  1.6*imbalance + 0.25*both_high - 0.25*(-closeness)

base_scores = np.stack([s_min, s_avg, s_max], axis=0)  # (3,n,n)

def compute_softmix(tau):
    S = tau * base_scores
    W = softmax(S, axis=0)  # (3,n,n)
    Z_soft = W[0]*Z_min + W[1]*Z_avg + W[2]*Z_max
    return Z_soft, W

# ---- Tau schedule (oscillation => "wing flapping") ----
frames = 120
tau_center = 8.0
tau_amp = 10.0
tau_values = tau_center + tau_amp * np.sin(np.linspace(0, 2*np.pi, frames))

# ---- Figure: 3D surface + 2D heatmap ----
fig = plt.figure(figsize=(12, 6))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax2d = fig.add_subplot(1, 2, 2)

weight_to_show = 2  # 0=min, 1=avg, 2=max

# Initialize
Z0, W0 = compute_softmix(float(tau_values[0]))

# 3D initial plot
ax3d.plot_surface(X, Y, Z0, rstride=3, cstride=3, alpha=0.95)
ax3d.set_title(f"Softmax mix surface (tau={tau_values[0]:.2f})")
ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("Z_soft")
ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 1); ax3d.set_zlim(0, 1)

# 2D initial heatmap
im = ax2d.imshow(W0[weight_to_show], origin="lower", extent=[0,1,0,1], aspect="equal")
ax2d.set_title(["W_min", "W_avg", "W_max"][weight_to_show] + " heatmap")
ax2d.set_xlabel("x"); ax2d.set_ylabel("y")
cbar = fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
cbar.set_label("weight")

plt.tight_layout()

def update(frame_idx):
    tau = float(tau_values[frame_idx])
    Z, W = compute_softmix(tau)

    # Robust clear for 3D axis across matplotlib versions
    ax3d.cla()
    ax3d.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.95)

    ax3d.set_title(f"Softmax mix surface (tau={tau:.2f})")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("Z_soft")
    ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 1); ax3d.set_zlim(0, 1)

    # Update heatmap
    im.set_data(W[weight_to_show])
    im.set_clim(0.0, 1.0)

    return []

anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)

# ---- Save as GIF ----
out_gif = "softmax_mix_tau_flap.gif"
anim.save(out_gif, writer=PillowWriter(fps=20))
print("Saved:", out_gif)

plt.show()
