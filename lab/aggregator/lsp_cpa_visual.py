import numpy as np
import matplotlib.pyplot as plt

W = np.linspace(0.05, 0.98, 300)
xs = [0.25, 0.5, 0.75]

# AG version
def ag_penalty_rel(W):
    return -100 * (1 - np.sqrt(W))

def ag_reward_rel(x, W):
    return 100 * (np.sqrt(W + (1 - W) / x) - 1)

# AH version
def ah_penalty_rel(W):
    return -100 * ((1 - W) / (1 + W))

def ah_reward_rel(x, W):
    return 100 * ((1 - W) * (1 - x)) / (x * (W + 1) + (1 - W))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# AG plot
ax = axes[0]
for x in xs:
    ax.plot(W, ag_reward_rel(x, W), linewidth=2)

ax.plot(W, ag_penalty_rel(W), linewidth=2)

ax.set_title("Penalty (-) and reward (+) for the AG version of CPA")
ax.set_xlabel("Weight W1")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.set_ylabel("Penalty [-%] and reward [+%]")
ax.set_xlim(0, 1)
ax.set_ylim(-80, 100)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(-80, 110, 10))
ax.grid(True)

# AH plot
ax = axes[1]
for x in xs:
    ax.plot(W, ah_reward_rel(x, W), linewidth=2)

ax.plot(W, ah_penalty_rel(W), linewidth=2)

ax.set_title("Penalty (-) and reward (+) for the AH version of CPA")
ax.set_xlabel("Weight W1")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.set_ylabel("Penalty [-%] and reward [+%]")
ax.set_xlim(0, 1)
ax.set_ylim(-100, 60)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(-100, 70, 10))
ax.grid(True)

plt.tight_layout()
plt.show()