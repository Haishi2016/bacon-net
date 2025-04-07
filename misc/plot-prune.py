import matplotlib.pyplot as plt

# Accuracy data for five models across pruning levels
data = [
    [94.93, 94.93, 94.93, 94.93, 82.54],  # Model 1
    [94.35, 94.35, 94.35, 87.79, 82.54],  # Model 2
    [94.01, 94.01, 94.01, 88.60, 83.62],  # Model 3
    [94.60, 94.60, 94.60, 94.60, 83.46],  # Model 4
    [95.26, 95.26, 95.26, 92.10, 82.88]   # Model 5
]

# X-axis: number of pruned features
x = [1, 2, 3, 4, 5]

# Plot setup
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D', 'v']
labels = [f'Model {i+1}' for i in range(5)]

for i, series in enumerate(data):
    plt.plot(x, series, label=labels[i], marker=markers[i])

# Vertical line at third data point (i.e., x=3)
plt.axvline(x=3, color='gray', linestyle='dotted', linewidth=5)

# Labels and legend
plt.xlabel("Pruned Features")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy vs. Pruned Features")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
