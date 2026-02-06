"""
Animated 3D plot of the LSP Softmax Aggregator.

The aggregator combines 5 operators with softmax-weighted mixing:
    A0(x,y) = x*y           (product)
    A1(x,y) = min(x,y)      (minimum)
    A2(x,y) = (x+y)/2       (average)
    A3(x,y) = max(x,y)      (maximum)
    A4(x,y) = x+y-x*y       (probabilistic sum)

Weights are computed as: w = softmax(-|a - center_i|^2 / tau)

Two animation modes:
  1. Animate andness (a) with selectable tau
  2. Animate tau with selectable andness

Usage:
    python plot_softmax_agg_animated.py --mode andness --tau 0.1
    python plot_softmax_agg_animated.py --mode tau --andness 0.75
    python plot_softmax_agg_animated.py --mode andness --save softmax_agg.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import argparse


# Default evenly-spaced centers (matching LspSoftmaxAggregator)
DEFAULT_CENTERS = np.array([1.5, 1.0, 0.5, 0.0, -0.5])
OPERATOR_NAMES = ["x·y", "min", "avg", "max", "x+y-xy"]


def compute_weights(a, centers, tau, eps=1e-10):
    """
    Compute softmax weights based on distance from andness to centers.
    
    w_i = softmax(-|a - center_i|^2 / tau)
    """
    distances = -((a - centers) ** 2)
    # Softmax with temperature
    exp_d = np.exp(distances / max(tau, eps))
    return exp_d / (exp_d.sum() + eps)


def softmax_aggregator(x, y, a, tau, centers=None, eps=1e-10):
    """
    Compute the softmax-weighted aggregation of 5 operators.
    
    Args:
        x, y: Input values in [0, 1] (can be arrays)
        a: Andness parameter (determines operator mixing)
        tau: Temperature (lower = sharper selection)
        centers: Operator centers (default: [1.5, 1.0, 0.5, 0.0, -0.5])
        
    Returns:
        z: Aggregated output in [0, 1]
    """
    if centers is None:
        centers = DEFAULT_CENTERS
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Compute weights from andness
    w = compute_weights(a, centers, tau, eps)
    
    # Compute all 5 operators
    ops = np.stack([
        x * y,                    # A0: product
        np.minimum(x, y),         # A1: min
        (x + y) / 2.0,            # A2: avg
        np.maximum(x, y),         # A3: max
        x + y - x * y,            # A4: prob sum
    ], axis=0)
    
    # Weighted sum: w is [5], ops is [5, ...]
    # Need to broadcast w to match ops shape
    w_broadcast = w.reshape([5] + [1] * (ops.ndim - 1))
    z = (w_broadcast * ops).sum(axis=0)
    
    return np.clip(z, 0.0, 1.0)


def get_operator_name(a, centers=None):
    """Get the dominant operator name for given andness."""
    if centers is None:
        centers = DEFAULT_CENTERS
    
    # Find closest center
    idx = np.argmin(np.abs(a - centers))
    return OPERATOR_NAMES[idx]


def format_weights(w):
    """Format weights for display."""
    parts = [f"{OPERATOR_NAMES[i]}:{w[i]:.2f}" for i in range(len(w))]
    return "  ".join(parts)


class AnimatedSoftmaxAggregator:
    """Animated 3D plot of LSP Softmax Aggregator."""
    
    def __init__(self, mode='andness', fixed_value=0.1, centers=None, 
                 resolution=35, fps=30):
        """
        Args:
            mode: 'andness' to animate a with fixed tau, 'tau' to animate tau with fixed a
            fixed_value: tau if mode='andness', a if mode='tau'
            centers: Operator centers (default: [1.5, 1.0, 0.5, 0.0, -0.5])
            resolution: Grid resolution
            fps: Frames per second
        """
        self.mode = mode
        self.fixed_value = fixed_value
        self.centers = centers if centers is not None else DEFAULT_CENTERS
        self.resolution = resolution
        self.fps = fps
        
        # Create meshgrid
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Animation parameters
        if mode == 'andness':
            # Sweep andness from min center to max center and back
            a_min, a_max = self.centers.min(), self.centers.max()
            self.param_range = np.concatenate([
                np.linspace(a_min, a_max, 80),
                np.linspace(a_max, a_min, 80)
            ])
            self.param_name = 'a'
            self.fixed_name = 'τ'
        else:
            # Sweep tau from sharp to soft and back
            self.param_range = np.concatenate([
                np.linspace(0.01, 1.0, 60),
                np.linspace(1.0, 0.01, 60)
            ])
            self.param_name = 'τ'
            self.fixed_name = 'a'
        
        self.setup_figure()
    
    def setup_figure(self):
        """Set up the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25, top=0.9)
        
        # Initial surface
        if self.mode == 'andness':
            Z = softmax_aggregator(self.X, self.Y, self.param_range[0], 
                                   self.fixed_value, self.centers)
        else:
            Z = softmax_aggregator(self.X, self.Y, self.fixed_value,
                                   self.param_range[0], self.centers)
        
        self.surf = self.ax.plot_surface(
            self.X, self.Y, Z,
            cmap='viridis',
            edgecolor='none',
            alpha=0.9
        )
        
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('y', fontsize=12)
        self.ax.set_zlabel('z', fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        self.ax.view_init(elev=25, azim=45)
        
        # Store current view for preservation during animation
        self.current_elev = 25
        self.current_azim = 45
        
        # Add slider for fixed parameter
        slider_ax = plt.axes([0.2, 0.08, 0.6, 0.03])
        if self.mode == 'andness':
            self.slider = Slider(slider_ax, 'τ (temperature)', 0.01, 1.0,
                                valinit=self.fixed_value, valstep=0.01)
        else:
            a_min, a_max = self.centers.min(), self.centers.max()
            self.slider = Slider(slider_ax, 'a (andness)', a_min, a_max,
                                valinit=self.fixed_value, valstep=0.05)
        
        self.slider.on_changed(self.on_slider_change)
        
        # Progress bar
        self.progress_ax = plt.axes([0.2, 0.13, 0.6, 0.02])
        self.progress_ax.set_xlim(0, 1)
        self.progress_ax.set_ylim(0, 1)
        self.progress_ax.set_xticks([])
        self.progress_ax.set_yticks([])
        
        # Weight display
        self.weight_ax = plt.axes([0.1, 0.02, 0.8, 0.04])
        self.weight_ax.set_xlim(0, 1)
        self.weight_ax.set_ylim(0, 1)
        self.weight_ax.axis('off')
        self.weight_text = self.weight_ax.text(0.5, 0.5, '', ha='center', 
                                                va='center', fontsize=10,
                                                family='monospace')
        
        # Center markers on progress bar
        if self.mode == 'andness':
            a_min, a_max = self.centers.min(), self.centers.max()
            for c in self.centers:
                pos = (c - a_min) / (a_max - a_min)
                self.progress_ax.axvline(pos, color='red', linewidth=2, alpha=0.7)
    
    def on_slider_change(self, val):
        """Handle slider changes."""
        self.fixed_value = val
    
    def update(self, frame):
        """Update function for animation."""
        # Save current view angle
        self.current_elev = self.ax.elev
        self.current_azim = self.ax.azim
        
        self.ax.clear()
        
        param = self.param_range[frame]
        
        if self.mode == 'andness':
            a = param
            tau = self.fixed_value
        else:
            a = self.fixed_value
            tau = param
        
        Z = softmax_aggregator(self.X, self.Y, a, tau, self.centers)
        w = compute_weights(a, self.centers, tau)
        
        dominant_idx = np.argmax(w)
        dominant_name = OPERATOR_NAMES[dominant_idx]
        
        self.ax.plot_surface(
            self.X, self.Y, Z,
            cmap='viridis',
            edgecolor='none',
            alpha=0.9
        )
        
        # Contour at bottom
        self.ax.contour(self.X, self.Y, Z, zdir='z', offset=0,
                       cmap='viridis', alpha=0.3, levels=10)
        
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('y', fontsize=12)
        self.ax.set_zlabel('z', fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        
        title = f'Softmax LSP Aggregator (a={a:.2f}, τ={tau:.3f})\nDominant: {dominant_name} ({w[dominant_idx]:.1%})'
        self.ax.set_title(title, fontsize=14)
        
        # Restore view angle
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
        
        # Update progress bar
        progress = frame / len(self.param_range)
        for patch in self.progress_ax.patches:
            patch.remove()
        self.progress_ax.axvspan(0, progress, color='steelblue', alpha=0.7)
        
        if self.mode == 'andness':
            a_min, a_max = self.centers.min(), self.centers.max()
            self.progress_ax.set_title(f'a = {a:.2f}  (range: {a_min} to {a_max})', fontsize=10)
        else:
            self.progress_ax.set_title(f'τ = {tau:.3f}  (range: 0.01 to 1.0)', fontsize=10)
        
        # Update weight display with bar chart style
        weight_str = "Weights: "
        for i, (name, weight) in enumerate(zip(OPERATOR_NAMES, w)):
            bar = '█' * int(weight * 10)
            weight_str += f"{name}:{weight:.2f}{bar}  "
        self.weight_text.set_text(weight_str)
        
        return []
    
    def animate(self, save_path=None):
        """Run the animation."""
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.param_range),
            interval=1000 // self.fps,
            blit=False
        )
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=self.fps)
            else:
                anim.save(save_path, writer='ffmpeg', fps=self.fps)
            print("Saved!")
        else:
            plt.show()
        
        return anim


def plot_comparison(a=0.75, tau=0.1, resolution=40):
    """
    Plot all 5 individual operators alongside the mixed result.
    """
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute all operators
    ops = [
        X * Y,
        np.minimum(X, Y),
        (X + Y) / 2.0,
        np.maximum(X, Y),
        X + Y - X * Y,
    ]
    
    # Compute mixed result
    w = compute_weights(a, DEFAULT_CENTERS, tau)
    Z_mixed = softmax_aggregator(X, Y, a, tau)
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(16, 10))
    
    for i, (op, name) in enumerate(zip(ops, OPERATOR_NAMES)):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.plot_surface(X, Y, op, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title(f'{name}\nw={w[i]:.3f}', fontsize=11)
        ax.view_init(elev=25, azim=45)
    
    # Mixed result in last subplot
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.plot_surface(X, Y, Z_mixed, cmap='plasma', edgecolor='none', alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title(f'Mixed (a={a:.2f}, τ={tau:.2f})', fontsize=11)
    ax.view_init(elev=25, azim=45)
    
    plt.suptitle(f'LSP Softmax Aggregator: 5 Operators + Mixed Result\n'
                 f'Centers: {list(DEFAULT_CENTERS)}', fontsize=14)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Animated 3D plot of LSP Softmax Aggregator')
    parser.add_argument('--mode', type=str, choices=['andness', 'tau'], default='andness',
                        help='Animation mode: "andness" animates a with fixed τ, '
                             '"tau" animates τ with fixed a (default: andness)')
    parser.add_argument('--andness', '-a', type=float, default=0.75,
                        help='Fixed andness when mode=tau (default: 0.75)')
    parser.add_argument('--tau', '-t', type=float, default=0.1,
                        help='Fixed tau when mode=andness (default: 0.1)')
    parser.add_argument('--centers', type=float, nargs=5, default=None,
                        help='Custom operator centers (5 values)')
    parser.add_argument('--resolution', type=int, default=35,
                        help='Grid resolution (default: 35)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (.gif or .mp4)')
    parser.add_argument('--compare', action='store_true',
                        help='Show comparison of all operators (no animation)')
    
    args = parser.parse_args()
    
    centers = np.array(args.centers) if args.centers else None
    
    if args.compare:
        fig = plot_comparison(a=args.andness, tau=args.tau, resolution=args.resolution)
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.save}")
        else:
            plt.show()
        return
    
    if args.mode == 'andness':
        fixed_value = args.tau
        print(f"Animating andness (a) with fixed τ = {fixed_value}")
    else:
        fixed_value = args.andness
        print(f"Animating τ with fixed a = {fixed_value}")
    
    animator = AnimatedSoftmaxAggregator(
        mode=args.mode,
        fixed_value=fixed_value,
        centers=centers,
        resolution=args.resolution,
        fps=args.fps
    )
    
    animator.animate(save_path=args.save)


if __name__ == "__main__":
    main()
