"""
Interactive 3D plot of the GCD2 (Generalized Conjunction/Disjunction) function.

GCD2(x, y, w, α) is a continuous interpolation between:
  - α = 2:   drastic AND (1 if x=y=1, else 0)
  - α = 1:   product (x^w * y^(1-w))^r  where r depends on α
  - α = 0.5: weighted average (wx + (1-w)y)
  - α = 0:   max-like OR
  - α = -1:  drastic OR (0 if x=y=0, else 1)

Usage:
    python plot_gcd2.py [--alpha 0.5] [--w 0.5] [--interactive]
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def gcd2(x, y, w, alpha, eps=1e-10):
    """
    Compute GCD2(x, y, w, α) - Generalized Conjunction/Disjunction.
    
    Args:
        x, y: Input values in [0, 1] (can be arrays)
        w: Weight parameter in [0, 1], controls relative importance of x vs y
        alpha: Andness parameter in [-1, 2]
            α = 2:    drastic AND
            α = 1:    geometric mean region
            α = 0.5:  arithmetic mean (weighted average)
            α = 0:    geometric mean region (OR side)
            α = -1:   drastic OR
        eps: Small constant for numerical stability
        
    Returns:
        z: GCD2(x, y, w, α) in [0, 1]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Clamp inputs to avoid numerical issues
    x = np.clip(x, eps, 1.0 - eps)
    y = np.clip(y, eps, 1.0 - eps)
    
    # α = 2: drastic AND (1 if x=y=1, else 0)
    if alpha >= 2.0 - eps:
        return np.where((x > 1.0 - eps) & (y > 1.0 - eps), 1.0, 0.0)
    
    # α in [-1, 0.5): use duality with α' = 1 - α
    if alpha < 0.5:
        # z = 1 - GCD2(1-x, 1-y, w, 1-α)
        return 1.0 - gcd2(1.0 - x, 1.0 - y, w, 1.0 - alpha, eps)
    
    # α = 0.5: weighted arithmetic mean
    if abs(alpha - 0.5) < eps:
        return w * x + (1.0 - w) * y
    
    # α in (0.5, 0.75]: interpolation between arithmetic and geometric mean
    # α in (0.75, 2): pure geometric mean with varying exponent
    
    # Compute the exponent r = sqrt(3 / (2 - α)) - 1
    r = np.sqrt(3.0 / (2.0 - alpha)) - 1.0
    
    # Compute weighted geometric mean: (x^(2w) * y^(2(1-w)))^r
    # = exp(r * (2w * log(x) + 2(1-w) * log(y)))
    log_x = np.log(x + eps)
    log_y = np.log(y + eps)
    geom_part = np.exp(r * (2.0 * w * log_x + 2.0 * (1.0 - w) * log_y))
    
    if alpha >= 0.75:
        # α in [0.75, 2): pure geometric mean formula
        return geom_part
    else:
        # α in (0.5, 0.75): linear interpolation between arithmetic and geometric
        # z = (3 - 4α)(wx + (1-w)y) + (4α - 2) * geom_part
        arith_part = w * x + (1.0 - w) * y
        coef_arith = 3.0 - 4.0 * alpha  # goes from 1 (at α=0.5) to 0 (at α=0.75)
        coef_geom = 4.0 * alpha - 2.0   # goes from 0 (at α=0.5) to 1 (at α=0.75)
        return coef_arith * arith_part + coef_geom * geom_part


def plot_gcd2_3d(alpha=0.5, w=0.5, resolution=50, elevation=30, azimuth=45):
    """
    Create a 3D surface plot of GCD2(x, y, w, α).
    
    Args:
        alpha: Andness parameter
        w: Weight parameter
        resolution: Grid resolution
        elevation: View elevation angle
        azimuth: View azimuth angle
    """
    # Create meshgrid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute GCD2
    Z = gcd2(X, Y, w, alpha)
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # Add contour lines on the bottom
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
    
    # Labels
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z = GCD₂(x, y, w, α)', fontsize=12)
    
    # Title with parameters
    op_name = get_operator_name(alpha)
    ax.set_title(f'GCD₂(x, y, w={w:.2f}, α={alpha:.2f})\n{op_name}', fontsize=14)
    
    # Set view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='z')
    
    plt.tight_layout()
    return fig, ax


def get_operator_name(alpha):
    """Get a human-readable name for the operator at given alpha."""
    if alpha >= 1.9:
        return "Drastic AND (t-norm)"
    elif alpha >= 1.4:
        return "Strong AND (near product)"
    elif alpha >= 1.1:
        return "Product t-norm"
    elif alpha >= 0.9:
        return "Geometric mean (weak AND)"
    elif alpha >= 0.6:
        return "Near arithmetic mean"
    elif alpha >= 0.4:
        return "Arithmetic mean (neutral)"
    elif alpha >= 0.1:
        return "Near arithmetic mean (OR side)"
    elif alpha >= -0.1:
        return "Geometric mean (weak OR)"
    elif alpha >= -0.4:
        return "Probabilistic sum region"
    elif alpha >= -0.9:
        return "Strong OR"
    else:
        return "Drastic OR (t-conorm)"


def interactive_plot():
    """Create an interactive plot with sliders for alpha and w."""
    from matplotlib.widgets import Slider
    
    # Initial parameters
    init_alpha = 0.5
    init_w = 0.5
    resolution = 40
    
    # Create meshgrid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)
    
    # Initial plot
    Z = gcd2(X, Y, init_w, init_alpha)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    title = ax.set_title(f'GCD₂(x, y, w={init_w:.2f}, α={init_alpha:.2f})\n{get_operator_name(init_alpha)}', fontsize=14)
    
    # Create sliders
    ax_alpha = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_w = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    slider_alpha = Slider(ax_alpha, 'α (andness)', -1.0, 2.0, valinit=init_alpha, valstep=0.05)
    slider_w = Slider(ax_w, 'w (weight)', 0.0, 1.0, valinit=init_w, valstep=0.05)
    
    def update(val):
        alpha = slider_alpha.val
        w = slider_w.val
        
        ax.clear()
        Z = gcd2(X, Y, w, alpha)
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title(f'GCD₂(x, y, w={w:.2f}, α={alpha:.2f})\n{get_operator_name(alpha)}', fontsize=14)
        
        fig.canvas.draw_idle()
    
    slider_alpha.on_changed(update)
    slider_w.on_changed(update)
    
    plt.show()


def plot_alpha_sweep(w=0.5, alphas=None, resolution=30):
    """
    Plot GCD2 for multiple alpha values in a grid.
    
    Args:
        w: Weight parameter
        alphas: List of alpha values to plot
        resolution: Grid resolution
    """
    if alphas is None:
        alphas = [-0.5, 0.0, 0.5, 1.0, 1.5]
    
    n_plots = len(alphas)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    for i, alpha in enumerate(alphas):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        Z = gcd2(X, Y, w, alpha)
        
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title(f'α = {alpha:.2f}\n{get_operator_name(alpha)}', fontsize=10)
        ax.view_init(elev=25, azim=45)
    
    plt.suptitle(f'GCD₂(x, y, w={w:.2f}, α) for various α values', fontsize=14)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot GCD2 function in 3D')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Andness parameter α in [-1, 2] (default: 0.5)')
    parser.add_argument('--w', type=float, default=0.5,
                        help='Weight parameter w in [0, 1] (default: 0.5)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Launch interactive plot with sliders')
    parser.add_argument('--sweep', '-s', action='store_true',
                        help='Plot multiple alpha values')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Grid resolution (default: 50)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of showing')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_plot()
    elif args.sweep:
        fig = plot_alpha_sweep(w=args.w, resolution=args.resolution)
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.save}")
        else:
            plt.show()
    else:
        fig, ax = plot_gcd2_3d(alpha=args.alpha, w=args.w, resolution=args.resolution)
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.save}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
