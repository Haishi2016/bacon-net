"""``bacon viz`` subcommand — animated aggregator surface visualization."""

import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Aggregator surface evaluators
# ---------------------------------------------------------------------------

def _softmax_surface(X, Y, a, tau=0.1):
    """LSP Softmax: 5 operators weighted by RBF distance to centres."""
    centres = np.array([1.5, 1.0, 0.5, 0.0, -0.5])
    d = -((a - centres) ** 2)
    w = np.exp(d / max(tau, 1e-10))
    w /= w.sum() + 1e-10
    ops = np.stack([
        X * Y,                 # product
        np.minimum(X, Y),      # min
        (X + Y) / 2.0,         # avg
        np.maximum(X, Y),      # max
        X + Y - X * Y,         # probabilistic sum
    ], axis=0)
    w_b = w.reshape([5] + [1] * (ops.ndim - 1))
    return np.clip((w_b * ops).sum(axis=0), 0, 1), w


def _min_max_surface(X, Y, a, tau=None):
    """Boolean Min/Max: linear interpolation between min and max."""
    a_c = np.clip(a, 0, 1)
    Z = a_c * np.minimum(X, Y) + (1 - a_c) * np.maximum(X, Y)
    w = np.array([a_c, 1 - a_c])
    return np.clip(Z, 0, 1), w


def _half_weight_surface(X, Y, a, tau=None):
    """Half-weight LSP aggregator (delegates to actual class)."""
    import torch
    from bacon.aggregators.lsp import HalfWeightAggregator
    agg = HalfWeightAggregator()
    x_f = torch.tensor(X.ravel(), dtype=torch.float32)
    y_f = torch.tensor(Y.ravel(), dtype=torch.float32)
    a_t = torch.tensor(float(a), dtype=torch.float32)
    w_t = [torch.tensor(0.5), torch.tensor(0.5)]
    with torch.no_grad():
        out = agg.aggregate_tensor([x_f, y_f], a_t, w_t)
    return np.clip(out.numpy().reshape(X.shape), 0, 1), None


def _full_weight_surface(X, Y, a, tau=None):
    """Full-weight LSP aggregator (delegates to actual class)."""
    import torch
    from bacon.aggregators.lsp import FullWeightAggregator
    agg = FullWeightAggregator()
    x_f = torch.tensor(X.ravel(), dtype=torch.float32)
    y_f = torch.tensor(Y.ravel(), dtype=torch.float32)
    a_t = torch.tensor(float(a), dtype=torch.float32)
    w_t = [torch.tensor(0.5), torch.tensor(0.5)]
    with torch.no_grad():
        out = agg.aggregate_tensor([x_f, y_f], a_t, w_t)
    return np.clip(out.numpy().reshape(X.shape), 0, 1), None


def _gl_generic_surface(X, Y, a, tau=None):
    """GL Generic: piecewise-linear interpolation between anchor functions."""
    eps = 1e-7
    anchors = [
        ('prob_sum',  -1 / 12,     lambda x, y: x + y - x * y),
        ('max',        0.0,        lambda x, y: np.maximum(x, y)),
        ('quadratic',  0.375,      lambda x, y: np.sqrt((x**2 + y**2) / 2)),
        ('mean',       0.5,        lambda x, y: (x + y) / 2),
        ('geometric',  0.625,      lambda x, y: np.sqrt(np.maximum(x * y, 0) + eps)),
        ('harmonic',   0.75,       lambda x, y: 2 * x * y / (x + y + eps)),
        ('min',        1.0,        lambda x, y: np.minimum(x, y)),
        ('product',    1 + 1 / 12, lambda x, y: x * y),
    ]
    a_lo, a_hi = anchors[0][1], anchors[-1][1]
    a_c = max(a_lo, min(a_hi, a))
    # Find bracketing pair
    for i in range(len(anchors) - 1):
        if a_c <= anchors[i + 1][1] + 1e-12:
            _, al, fn_l = anchors[i]
            _, ah, fn_h = anchors[i + 1]
            t = (a_c - al) / (ah - al) if abs(ah - al) > 1e-12 else 0.5
            Z = (1 - t) * fn_l(X, Y) + t * fn_h(X, Y)
            # Compute weight-like info: fraction toward higher-andness anchor
            w = np.zeros(len(anchors))
            w[i] = 1 - t
            w[i + 1] = t
            return np.clip(Z, 0, 1), w
    return np.clip(anchors[-1][2](X, Y), 0, 1), None


# ---------------------------------------------------------------------------
# Aggregator configuration registry
# ---------------------------------------------------------------------------

_AGGREGATORS = {
    'lsp.softmax': {
        'label': 'LSP Softmax Aggregator',
        'andness_range': (-0.5, 1.5),
        'operators': ['x\u00b7y', 'min', 'avg', 'max', 'x+y-xy'],
        'centres': [1.5, 1.0, 0.5, 0.0, -0.5],
        'surface_fn': _softmax_surface,
        'has_tau': True,
    },
    'lsp.half_weight': {
        'label': 'LSP Half-Weight Aggregator',
        'andness_range': (-1.0, 2.0),
        'operators': None,
        'centres': None,
        'surface_fn': _half_weight_surface,
        'has_tau': False,
    },
    'lsp.full_weight': {
        'label': 'LSP Full-Weight Aggregator',
        'andness_range': (-1.0, 2.0),
        'operators': None,
        'centres': None,
        'surface_fn': _full_weight_surface,
        'has_tau': False,
    },
    'bool.min_max': {
        'label': 'Boolean Min/Max Aggregator',
        'andness_range': (0.0, 1.0),
        'operators': ['min (AND)', 'max (OR)'],
        'centres': [1.0, 0.0],
        'surface_fn': _min_max_surface,
        'has_tau': False,
    },
    'gl.generic': {
        'label': 'Generic GL Aggregator',
        'andness_range': (-1 / 12, 1 + 1 / 12),
        'operators': ['prob_sum', 'max', 'quadratic', 'mean',
                      'geometric', 'harmonic', 'min', 'product'],
        'centres': [-1 / 12, 0.0, 0.375, 0.5, 0.625, 0.75, 1.0, 1 + 1 / 12],
        'surface_fn': _gl_generic_surface,
        'has_tau': False,
    },
}


# ---------------------------------------------------------------------------
# Animated 3-D surface plotter
# ---------------------------------------------------------------------------

class AggregatorAnimator:
    """Animated 3-D surface that sweeps andness for any registered aggregator."""

    def __init__(self, aggregator, mode='andness', tau=0.1,
                 fixed_andness=None, resolution=35, fps=30):
        cfg = _AGGREGATORS[aggregator]
        self.cfg = cfg
        self.aggregator = aggregator
        self.surface_fn = cfg['surface_fn']
        self.tau = tau
        self.resolution = resolution
        self.fps = fps
        self.mode = mode

        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        self.X, self.Y = np.meshgrid(x, y)

        a_min, a_max = cfg['andness_range']

        if mode == 'andness':
            self.param_range = np.concatenate([
                np.linspace(a_min, a_max, 80),
                np.linspace(a_max, a_min, 80),
            ])
            self.param_name = 'a'
            self.fixed_name = '\u03c4'
        else:
            # tau sweep (only meaningful for softmax)
            self.param_range = np.concatenate([
                np.linspace(0.01, 1.0, 60),
                np.linspace(1.0, 0.01, 60),
            ])
            self.param_name = '\u03c4'
            self.fixed_name = 'a'
            self.fixed_andness = fixed_andness if fixed_andness is not None else (a_min + a_max) / 2

        self._setup_figure(a_min, a_max)

    # ------------------------------------------------------------------ #

    def _setup_figure(self, a_min, a_max):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25, top=0.9)

        # Initial surface
        a0 = self.param_range[0]
        if self.mode == 'andness':
            Z, _ = self.surface_fn(self.X, self.Y, a0, self.tau)
        else:
            Z, _ = self.surface_fn(self.X, self.Y, self.fixed_andness, a0)

        self.ax.plot_surface(self.X, self.Y, Z, cmap='viridis',
                             edgecolor='none', alpha=0.9)
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('y', fontsize=12)
        self.ax.set_zlabel('z', fontsize=12)
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.ax.set_zlim(0, 1)
        self.ax.view_init(elev=25, azim=45)
        self.current_elev = 25
        self.current_azim = 45

        # Slider for the fixed parameter
        slider_ax = plt.axes([0.2, 0.08, 0.6, 0.03])
        if self.mode == 'andness' and self.cfg['has_tau']:
            self.slider = Slider(slider_ax, '\u03c4 (temperature)', 0.01, 1.0,
                                 valinit=self.tau, valstep=0.01)
            self.slider.on_changed(self._on_tau_change)
        elif self.mode == 'tau':
            self.slider = Slider(slider_ax, 'a (andness)', a_min, a_max,
                                 valinit=self.fixed_andness, valstep=0.05)
            self.slider.on_changed(self._on_andness_change)
        else:
            slider_ax.set_visible(False)
            self.slider = None

        # Progress bar
        self.progress_ax = plt.axes([0.2, 0.13, 0.6, 0.02])
        self.progress_ax.set_xlim(0, 1); self.progress_ax.set_ylim(0, 1)
        self.progress_ax.set_xticks([]); self.progress_ax.set_yticks([])

        # Centre markers
        if self.mode == 'andness' and self.cfg.get('centres'):
            for c in self.cfg['centres']:
                pos = (c - a_min) / (a_max - a_min) if a_max != a_min else 0.5
                self.progress_ax.axvline(pos, color='red', linewidth=2, alpha=0.7)

        # Weight text
        self.weight_ax = plt.axes([0.1, 0.02, 0.8, 0.04])
        self.weight_ax.set_xlim(0, 1); self.weight_ax.set_ylim(0, 1)
        self.weight_ax.axis('off')
        self.weight_text = self.weight_ax.text(
            0.5, 0.5, '', ha='center', va='center',
            fontsize=10, family='monospace',
        )

    def _on_tau_change(self, val):
        self.tau = val

    def _on_andness_change(self, val):
        self.fixed_andness = val

    # ------------------------------------------------------------------ #

    def _format_weights(self, w):
        ops = self.cfg.get('operators')
        if w is None or ops is None:
            return ''
        parts = []
        for name, wi in zip(ops, w):
            if wi > 0.005:
                bar = '\u2588' * int(wi * 10)
                parts.append(f'{name}:{wi:.2f}{bar}')
        return '  '.join(parts)

    def update(self, frame):
        import matplotlib.pyplot as plt          # noqa: F811

        self.current_elev = self.ax.elev
        self.current_azim = self.ax.azim
        self.ax.clear()

        p = self.param_range[frame]
        if self.mode == 'andness':
            a, tau = p, self.tau
        else:
            a, tau = self.fixed_andness, p

        Z, w = self.surface_fn(self.X, self.Y, a, tau)

        self.ax.plot_surface(self.X, self.Y, Z, cmap='viridis',
                             edgecolor='none', alpha=0.9)
        self.ax.contour(self.X, self.Y, Z, zdir='z', offset=0,
                        cmap='viridis', alpha=0.3, levels=10)
        self.ax.set_xlabel('x'); self.ax.set_ylabel('y'); self.ax.set_zlabel('z')
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.ax.set_zlim(0, 1)
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)

        title = f'{self.cfg["label"]}  (a={a:.3f}'
        if self.cfg['has_tau']:
            title += f', \u03c4={tau:.3f}'
        title += ')'
        self.ax.set_title(title, fontsize=14)

        # Progress bar
        progress = frame / len(self.param_range)
        for patch in self.progress_ax.patches:
            patch.remove()
        self.progress_ax.axvspan(0, progress, color='steelblue', alpha=0.7)
        a_min, a_max = self.cfg['andness_range']
        if self.mode == 'andness':
            self.progress_ax.set_title(
                f'a = {a:.3f}  (range: {a_min:.3f} to {a_max:.3f})', fontsize=10)
        else:
            self.progress_ax.set_title(
                f'\u03c4 = {tau:.3f}  (range: 0.01 to 1.0)', fontsize=10)

        self.weight_text.set_text(self._format_weights(w))
        return []

    # ------------------------------------------------------------------ #

    def animate(self, save_path=None):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(
            self.fig, self.update,
            frames=len(self.param_range),
            interval=1000 // self.fps,
            blit=False,
        )

        if save_path:
            print(f'Saving animation to {save_path} ...')
            writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
            anim.save(save_path, writer=writer, fps=self.fps)
            print('Done.')
        else:
            plt.show()
        return anim


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def add_viz_parser(subparsers):
    """Register the ``viz`` subcommand."""
    agg_names = list(_AGGREGATORS.keys())

    parser = subparsers.add_parser(
        'viz',
        help='Animated 3-D aggregator surface visualization',
        description='Animate the aggregation surface z = agg(x, y) as '
                    'andness sweeps across the full range for a given '
                    'aggregator type.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )

    parser.add_argument(
        'aggregator',
        choices=agg_names,
        metavar='AGGREGATOR',
        help='Aggregator type: ' + ', '.join(agg_names),
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['andness', 'tau'],
        default='andness',
        help='Animate andness (default) or tau (softmax only)',
    )
    parser.add_argument(
        '--tau', '-t', type=float, default=0.1,
        help='Fixed tau when mode=andness (default: 0.1)',
    )
    parser.add_argument(
        '--andness', '-a', type=float, default=None,
        help='Fixed andness when mode=tau',
    )
    parser.add_argument(
        '--resolution', type=int, default=35,
        help='Grid resolution (default: 35)',
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Frames per second (default: 30)',
    )
    parser.add_argument(
        '--save', '-s', type=str, default=None,
        help='Save animation to file (.gif or .mp4)',
    )

    parser.set_defaults(command='viz', func=_run_viz)


def _build_epilog():
    lines = ['Available aggregators and their andness ranges:\n']
    for name, cfg in _AGGREGATORS.items():
        lo, hi = cfg['andness_range']
        lines.append(f'  {name:20s}  [{lo:+.3f}, {hi:+.3f}]  {cfg["label"]}')
    return '\n'.join(lines)


def _run_viz(args):
    cfg = _AGGREGATORS[args.aggregator]
    if args.mode == 'tau' and not cfg['has_tau']:
        print(f'Tau sweep is only supported for lsp.softmax.')
        return 1

    a_min, a_max = cfg['andness_range']
    print(f'Aggregator : {cfg["label"]}')
    print(f'Andness    : [{a_min:+.4f}, {a_max:+.4f}]')
    if cfg['has_tau']:
        print(f'Tau        : {args.tau}')

    animator = AggregatorAnimator(
        aggregator=args.aggregator,
        mode=args.mode,
        tau=args.tau,
        fixed_andness=args.andness,
        resolution=args.resolution,
        fps=args.fps,
    )
    animator.animate(save_path=args.save)
    return 0
