"""
Test and visualize the difference between hard blocks and soft boundaries
in hierarchical permutation initialization.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from bacon.inputToLeafSinkhorn import inputToLeafSinkhorn

def visualize_initialization(n=12, group_size=3, coarse_perm=None, 
                            bleed_ratio=0.0, bleed_decay=2.0, title=""):
    """Visualize how the permutation matrix is initialized."""
    
    if coarse_perm is None:
        # Default: identity permutation
        k = (n + group_size - 1) // group_size
        coarse_perm = list(range(k))
    
    # Create inputToLeafSinkhorn instance
    layer = inputToLeafSinkhorn(
        num_inputs=n,
        num_leaves=n,
        temperature=1.0,
        sinkhorn_iters=20,
        use_gumbel=False
    )
    
    # Initialize with hierarchical permutation
    layer.initialize_from_coarse_permutation(
        coarse_perm=coarse_perm,
        group_size=group_size,
        block_std=0.5,
        bleed_ratio=bleed_ratio,
        bleed_decay=bleed_decay
    )
    
    # Get the logits
    logits = layer.logits.detach().cpu().numpy()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Raw logits (heatmap)
    im1 = ax1.imshow(logits, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    ax1.set_title(f'{title}\nRaw Logits')
    ax1.set_xlabel('Column (to)')
    ax1.set_ylabel('Row (from)')
    plt.colorbar(im1, ax=ax1)
    
    # Add grid lines for groups
    k = len(coarse_perm)
    for i in range(1, k):
        pos = i * group_size - 0.5
        ax1.axhline(pos, color='white', linewidth=2, alpha=0.7)
        ax1.axvline(pos, color='white', linewidth=2, alpha=0.7)
    
    # Plot 2: After softmax (probabilities)
    P = torch.softmax(layer.logits / layer.temperature, dim=1).detach().cpu().numpy()
    im2 = ax2.imshow(P, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title(f'{title}\nRow-wise Softmax Probabilities')
    ax2.set_xlabel('Column (to)')
    ax2.set_ylabel('Row (from)')
    plt.colorbar(im2, ax=ax2)
    
    # Add grid lines for groups
    for i in range(1, k):
        pos = i * group_size - 0.5
        ax2.axhline(pos, color='white', linewidth=2, alpha=0.7)
        ax2.axvline(pos, color='white', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n{title}")
    print("=" * 60)
    print(f"Coarse permutation: {coarse_perm}")
    print(f"Logits - min: {logits.min():.3f}, max: {logits.max():.3f}, std: {logits.std():.3f}")
    
    # Check entropy per row (how spread out the distribution is)
    entropy_per_row = -(P * np.log(P + 1e-10)).sum(axis=1)
    print(f"Average row entropy: {entropy_per_row.mean():.3f} (max={np.log(n):.3f})")
    
    # Check mass concentration in intended blocks
    total_mass_in_blocks = 0
    for i, target_col_group in enumerate(coarse_perm):
        row_start = i * group_size
        row_end = min((i + 1) * group_size, n)
        col_start = target_col_group * group_size
        col_end = min((target_col_group + 1) * group_size, n)
        
        block_mass = P[row_start:row_end, col_start:col_end].sum()
        total_mass_in_blocks += block_mass
    
    avg_mass_in_intended_blocks = total_mass_in_blocks / len(coarse_perm)
    print(f"Average mass in intended blocks: {avg_mass_in_intended_blocks:.3f} (out of {group_size:.1f})")
    print(f"Concentration ratio: {avg_mass_in_intended_blocks / group_size * 100:.1f}%")
    
    return fig


def main():
    """Compare different bleeding configurations."""
    
    n = 12  # 12 features
    group_size = 3  # 12/3 = 4 groups
    coarse_perm = [2, 0, 3, 1]  # Non-identity permutation for visualization
    
    print("\n" + "=" * 70)
    print("HIERARCHICAL PERMUTATION: HARD BLOCKS VS SOFT BOUNDARIES")
    print("=" * 70)
    print(f"\nConfiguration: {n} features, {group_size} per group → {len(coarse_perm)}×{len(coarse_perm)} coarse matrix")
    print(f"Coarse permutation: {coarse_perm}")
    print()
    
    # Test 1: Hard blocks (original)
    fig1 = visualize_initialization(
        n=n, group_size=group_size, coarse_perm=coarse_perm,
        bleed_ratio=0.0,
        title="Hard Blocks (bleed_ratio=0.0)"
    )
    
    # Test 2: Soft boundaries (10% bleed)
    fig2 = visualize_initialization(
        n=n, group_size=group_size, coarse_perm=coarse_perm,
        bleed_ratio=0.1, bleed_decay=2.0,
        title="Soft Boundaries (bleed_ratio=0.1, decay=2.0)"
    )
    
    # Test 3: Very soft (30% bleed, slower decay)
    fig3 = visualize_initialization(
        n=n, group_size=group_size, coarse_perm=coarse_perm,
        bleed_ratio=0.3, bleed_decay=1.5,
        title="Very Soft (bleed_ratio=0.3, decay=1.5)"
    )
    
    # Test 4: Extreme soft (50% bleed)
    fig4 = visualize_initialization(
        n=n, group_size=group_size, coarse_perm=coarse_perm,
        bleed_ratio=0.5, bleed_decay=1.0,
        title="Extreme Soft (bleed_ratio=0.5, decay=1.0)"
    )
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Hard Blocks (bleed_ratio=0.0):
  - Strong bias toward coarse structure
  - Fast convergence if coarse permutation is correct
  - Can't recover easily if coarse structure is wrong
  - Best for: Known good coarse structure, fast convergence

Soft Boundaries (bleed_ratio=0.1):
  - Maintains strong bias but allows flexibility
  - Adjacent blocks get weak signal (exploration)
  - Can recover from suboptimal coarse structure
  - Best for: Default choice, balanced exploration/exploitation

Very Soft (bleed_ratio=0.3):
  - More exploratory
  - Features can migrate across group boundaries
  - Slower initial convergence
  - Best for: Uncertain coarse structure, complex relationships

Extreme Soft (bleed_ratio=0.5):
  - Very weak coarse bias
  - Almost like random initialization with slight structure
  - Much slower convergence
  - Best for: When coarse structure is just a guess
    """)


if __name__ == '__main__':
    main()
