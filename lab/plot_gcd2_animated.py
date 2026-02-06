"""
Animated 3D plot of the GCD2 function with sweeping alpha or w.

Two animation modes:
  1. Animate alpha from -1 to 2 with selectable weight w
  2. Animate w from 0 to 1 with selectable alpha

Usage:
    python plot_gcd2_animated.py --mode alpha --w 0.5
    python plot_gcd2_animated.py --mode weight --alpha 0.75
    python plot_gcd2_animated.py --mode alpha --save gcd2_alpha.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import argparse


def gcd2(x, y, w, alpha, eps=1e-10):
    """
    Compute GCD2(x, y, w, α) - Generalized Conjunction/Disjunction.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    x = np.clip(x, eps, 1.0 - eps)
    y = np.clip(y, eps, 1.0 - eps)
    
    if alpha >= 2.0 - eps:
        return np.where((x > 1.0 - eps) & (y > 1.0 - eps), 1.0, 0.0)
    
    if alpha < 0.5:
        return 1.0 - gcd2(1.0 - x, 1.0 - y, w, 1.0 - alpha, eps)
    
    if abs(alpha - 0.5) < eps:
        return w * x + (1.0 - w) * y
    
    r = np.sqrt(3.0 / (2.0 - alpha)) - 1.0
    log_x = np.log(x + eps)
    log_y = np.log(y + eps)
    geom_part = np.exp(r * (2.0 * w * log_x + 2.0 * (1.0 - w) * log_y))
    
    if alpha >= 0.75:
        return geom_part
    else:
        arith_part = w * x + (1.0 - w) * y
        coef_arith = 3.0 - 4.0 * alpha
        coef_geom = 4.0 * alpha - 2.0
        return coef_arith * arith_part + coef_geom * geom_part


def get_operator_name(alpha):
    """Get a human-readable name for the operator at given alpha."""
    if alpha >= 1.9:
        return "Drastic AND"
    elif alpha >= 1.4:
        return "Strong AND"
    elif alpha >= 1.1:
        return "Product (t-norm)"
    elif alpha >= 0.9:
        return "Weak AND"
    elif alpha >= 0.6:
        return "Near average"
    elif alpha >= 0.4:
        return "Arithmetic mean"
    elif alpha >= 0.1:
        return "Near average (OR)"
    elif alpha >= -0.1:
        return "Weak OR"
    elif alpha >= -0.4:
        return "Probabilistic sum"
    elif alpha >= -0.9:
        return "Strong OR"
    else:
        return "Drastic OR"


class AnimatedGCD2:
    """Animated 3D plot of GCD2 with parameter sweep."""
    
    def __init__(self, mode='alpha', fixed_value=0.5, resolution=35, fps=30):
        """
        Args:
            mode: 'alpha' to animate alpha with fixed w, 'weight' to animate w with fixed alpha
            fixed_value: The fixed parameter value (w if mode='alpha', alpha if mode='weight')
            resolution: Grid resolution for the surface
            fps: Frames per second for animation
        """
        self.mode = mode
        self.fixed_value = fixed_value
        self.resolution = resolution
        self.fps = fps
        
        # Create meshgrid
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Animation parameters
        if mode == 'alpha':
            self.param_range = np.concatenate([
                np.linspace(-1, 2, 90),    # Forward
                np.linspace(2, -1, 90)     # Backward
            ])
            self.param_name = 'α'
            self.fixed_name = 'w'
        else:
            self.param_range = np.concatenate([
                np.linspace(0, 1, 60),     # Forward
                np.linspace(1, 0, 60)      # Backward
            ])
            self.param_name = 'w'
            self.fixed_name = 'α'
        
        self.setup_figure()
    
    def setup_figure(self):
        """Set up the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.2)
        
        # Initial surface
        if self.mode == 'alpha':
            Z = gcd2(self.X, self.Y, self.fixed_value, self.param_range[0])
        else:
            Z = gcd2(self.X, self.Y, self.param_range[0], self.fixed_value)
        
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
        self.ax.view_init(elev=25, azim=45)  # Set only once at init
        
        # Store current view for preservation during animation
        self.current_elev = 25
        self.current_azim = 45
        
        # Add slider for fixed parameter
        slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
        if self.mode == 'alpha':
            self.slider = Slider(slider_ax, 'w (weight)', 0.0, 1.0, 
                                valinit=self.fixed_value, valstep=0.05)
        else:
            self.slider = Slider(slider_ax, 'α (andness)', -1.0, 2.0, 
                                valinit=self.fixed_value, valstep=0.05)
        
        self.slider.on_changed(self.on_slider_change)
        
        # Progress bar axes
        self.progress_ax = plt.axes([0.2, 0.1, 0.6, 0.02])
        self.progress_ax.set_xlim(0, 1)
        self.progress_ax.set_ylim(0, 1)
        self.progress_ax.set_xticks([])
        self.progress_ax.set_yticks([])
        self.progress_bar = self.progress_ax.axvspan(0, 0, color='steelblue', alpha=0.7)
        
        if self.mode == 'alpha':
            self.progress_ax.set_title('α: -1 ← → 2', fontsize=10)
        else:
            self.progress_ax.set_title('w: 0 ← → 1', fontsize=10)
    
    def on_slider_change(self, val):
        """Handle slider value changes."""
        self.fixed_value = val
    
    def update(self, frame):
        """Update function for animation."""
        # Save current view angle before clearing (preserves mouse rotation)
        self.current_elev = self.ax.elev
        self.current_azim = self.ax.azim
        
        self.ax.clear()
        
        param = self.param_range[frame]
        
        if self.mode == 'alpha':
            alpha = param
            w = self.fixed_value
            Z = gcd2(self.X, self.Y, w, alpha)
            op_name = get_operator_name(alpha)
            title = f'GCD₂(x, y, w={w:.2f}, α={alpha:.2f})\n{op_name}'
        else:
            w = param
            alpha = self.fixed_value
            Z = gcd2(self.X, self.Y, w, alpha)
            op_name = get_operator_name(alpha)
            title = f'GCD₂(x, y, w={w:.2f}, α={alpha:.2f})\n{op_name}'
        
        self.ax.plot_surface(
            self.X, self.Y, Z,
            cmap='viridis',
            edgecolor='none',
            alpha=0.9
        )
        
        # Add contour at bottom
        self.ax.contour(self.X, self.Y, Z, zdir='z', offset=0, 
                       cmap='viridis', alpha=0.3, levels=10)
        
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('y', fontsize=12)
        self.ax.set_zlabel('z', fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        self.ax.set_title(title, fontsize=14)
        
        # Restore view angle (preserves mouse rotation)
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
        
        # Update progress bar
        progress = frame / len(self.param_range)
        for patch in self.progress_ax.patches:
            patch.remove()
        self.progress_ax.axvspan(0, progress, color='steelblue', alpha=0.7)
        
        # Show current animated parameter value
        if self.mode == 'alpha':
            self.progress_ax.set_title(f'α = {alpha:.2f}  (range: -1 to 2)', fontsize=10)
        else:
            self.progress_ax.set_title(f'w = {w:.2f}  (range: 0 to 1)', fontsize=10)
        
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
            print(f"Saved!")
        else:
            plt.show()
        
        return anim


def main():
    parser = argparse.ArgumentParser(description='Animated 3D plot of GCD2 function')
    parser.add_argument('--mode', type=str, choices=['alpha', 'weight'], default='alpha',
                        help='Animation mode: "alpha" animates α with fixed w, '
                             '"weight" animates w with fixed α (default: alpha)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='Fixed α value when mode=weight (default: 0.75)')
    parser.add_argument('--w', type=float, default=0.5,
                        help='Fixed w value when mode=alpha (default: 0.5)')
    parser.add_argument('--resolution', type=int, default=35,
                        help='Grid resolution (default: 35)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (.gif or .mp4)')
    
    args = parser.parse_args()
    
    if args.mode == 'alpha':
        fixed_value = args.w
        print(f"Animating α from -1 to 2 with fixed w = {fixed_value}")
    else:
        fixed_value = args.alpha
        print(f"Animating w from 0 to 1 with fixed α = {fixed_value}")
    
    animator = AnimatedGCD2(
        mode=args.mode,
        fixed_value=fixed_value,
        resolution=args.resolution,
        fps=args.fps
    )
    
    animator.animate(save_path=args.save)


if __name__ == "__main__":
    main()
