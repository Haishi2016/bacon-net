import numpy as np
import matplotlib.pyplot as plt
from math import lgamma

# Input sizes and test data
inputs = np.array([3, 5, 10, 20, 50, 100, 200])
data = np.array([
    [100,100,96,96,97,96,95],
    [100,100,96,np.nan,96,97,96],
    [100,100,96,97,94,95,95],
    [100,97,99,98,97,95,np.nan],
    [100,100,97,99,99,95,94],
    [100,100,96,94,97,94,94],
    [100,100,96,99,96,95,95],
    [100,np.nan,96,96,94,95,100],
    [100,100,98,94,95,96,100],
    [100,97,98,100,100,96,97]
], dtype=float)

# Replace NaNs with column means
data = np.where(np.isnan(data), np.nanmean(data, axis=0), data)

# Compute medians for the trend line
medians = np.median(data, axis=0)

# Compute log10(n!) safely (no overflow)
log10_fact = np.array([lgamma(n + 1) / np.log(10) for n in inputs])

# --- Plot setup ---
fig, ax1 = plt.subplots(figsize=(8,5))

# Boxplot for accuracy (with outliers)
boxprops = dict(facecolor='lightblue', color='navy', alpha=0.7)
medianprops = dict(color='navy', linewidth=1.5)
whiskerprops = dict(color='navy')
capprops = dict(color='navy')

bp = ax1.boxplot(
    [data[:, i] for i in range(len(inputs))],
    positions=inputs,
    widths=4,
    patch_artist=True,
    showfliers=True,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops
)

# Median trend line
ax1.plot(inputs, medians, '-o', color='navy', linewidth=1.5, label='Median accuracy')

# Axis and grid
ax1.set_xlabel('Input size (n)')
ax1.set_ylabel('Accuracy (%)', color='navy')
ax1.set_ylim(90, 102)
ax1.tick_params(axis='y', labelcolor='navy')
ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
ax1.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
ax1.minorticks_on()

# --- Second axis: log10(n!) curve ---
ax2 = ax1.twinx()
ax2.plot(inputs, log10_fact, '--o', color='brown', linewidth=2, label='Search space')
ax2.set_ylabel('Search space (log₁₀ n!)', color='brown')
ax2.tick_params(axis='y', labelcolor='brown')
ax2.set_ylim(0, 400)
ax2.yaxis.grid(True, which='major', linestyle=':', linewidth=0.3, alpha=0.5)

# Add annotation labels showing 10^xx scale
for x, y in list(zip(inputs, log10_fact))[3:]:
    ax2.text(
        x + 6, y-12,              # small offset above each point
        f"$10^{{{int(y)}}}$",     # round exponent
        color='brown',
        fontsize=8,
        ha='center'
    )

# --- Legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

# --- Title ---
plt.title('BACON Accuracy and Search-Space Growth vs Input Size')

plt.tight_layout()
plt.show()
