"""
Generic Graded Logic (GL) Aggregator

Implements the canonical aggregation form from the GL aggregator paper:

    A(x | c) = sum_i  alpha_i(psi(u, c)) * F_i(u),    u = II(x)

where:
    - x = (x1, ..., xn) are the original inputs in [0, 1]
    - II(x) is a learned row-stochastic coordinate transformation R
    - F_i(u) are anchor aggregation operators (min, mean, max, ...)
    - psi(u, c) is a feature mapping capturing relational input characteristics
    - alpha_i are weights (static, conditional, or value-dependent)
    - c is an optional external context variable

The architecture supports **N-ary** inputs and four weight modes:

    1. **Static**: fixed convex combination of anchors (classical MAT).
    2. **Composite**: coordinate transformation R enables composite operators
       like partial absorption to emerge from standard anchors.
    3. **Conditional**: weights depend on external context c via neural gating.
    4. **Value-dependent**: weights depend on input features psi(u) via neural
       gating, enabling joint-condition aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Optional, Any
from bacon.aggregators.base import AggregatorBase


# ---------------------------------------------------------------------------
# N-ary anchor operator library
#
# The six core anchors form the standard GL generalized-mean family used in
# the paper (Section 4.5.1): F = {min, H, G, A, Q, max}.
# Two extended anchors (product, prob_sum) are available for compatibility
# with t-norm / t-conorm aggregation but are NOT part of the paper's set.
# ---------------------------------------------------------------------------

def _n_min(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Minimum (pure conjunction)."""
    return u.min(dim=0).values


def _n_harmonic_mean(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Harmonic mean: N / sum(1/x_i).  Strong quasi-conjunction."""
    N = u.size(0)
    # Clamp inputs away from zero to prevent 1/x explosion and gradient blow-up.
    u_safe = u.clamp(min=eps)
    return N / (1.0 / u_safe).sum(dim=0)


def _n_geometric_mean(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Geometric mean: exp(mean(log(x_i))).  Medium quasi-conjunction.

    Uses log-space computation to avoid product underflow and provide
    numerically stable gradients.
    """
    u_safe = u.clamp(min=eps)
    return torch.exp(u_safe.log().mean(dim=0))


def _n_mean(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Arithmetic mean.  Neutral (andness = 0.5)."""
    return u.mean(dim=0)


def _n_quadratic_mean(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Quadratic (RMS) mean: sqrt(mean(x_i^2)).  Medium quasi-disjunction."""
    return (u ** 2).mean(dim=0).clamp(min=eps * eps).sqrt()


def _n_max(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Maximum (pure disjunction)."""
    return u.max(dim=0).values


# --- Extended anchors (not in the paper) ---

def _n_product(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Algebraic product (t-norm): prod(x_i).  Stronger than min."""
    return u.prod(dim=0)


def _n_prob_sum(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Probabilistic sum (t-conorm): 1 - prod(1 - x_i).  Stronger than max."""
    return 1.0 - (1.0 - u).prod(dim=0)


ANCHOR_FUNCTIONS = {
    # Core GL anchors (paper Section 4.5.1)
    'min':       _n_min,
    'harmonic':  _n_harmonic_mean,
    'geometric': _n_geometric_mean,
    'mean':      _n_mean,
    'quadratic': _n_quadratic_mean,
    'max':       _n_max,
    # Extended (t-norm / t-conorm)
    'product':   _n_product,
    'prob_sum':  _n_prob_sum,
}

# Andness values for each anchor.
# The six core anchors span the [0, 1] GL andness continuum:
#   min=1 > harmonic=3/4 > geometric=5/8 > mean=1/2 > quadratic=3/8 > max=0
# Extended anchors lie outside [0, 1].
ANCHOR_ANDNESS = {
    'min':       1.0,
    'harmonic':  0.75,
    'geometric': 0.625,
    'mean':      0.5,
    'quadratic': 0.375,
    'max':       0.0,
    'product':   1.0 + 1 / 12,   # ~1.083, stronger than min
    'prob_sum':  0.0 - 1 / 12,   # ~-0.083, stronger than max
}

# Number of summary features produced by psi
_PSI_DIM = 5


# ---------------------------------------------------------------------------
# Anchor interpolation helper
# ---------------------------------------------------------------------------

def _make_interpolated_fn(fn_a, fn_b, t: float):
    """Create a function that linearly interpolates between two anchors.

    Returns ``(1-t)*fn_a(u) + t*fn_b(u)``.
    """
    def _interp(u: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        return (1.0 - t) * fn_a(u, eps) + t * fn_b(u, eps)
    return _interp


def _expand_anchors_with_interpolation(
    names: list[str],
    fns: list,
    andness_vals: list[float],
    n_interp: int,
) -> tuple[list[str], list, list[float]]:
    """Insert *n_interp* linearly interpolated anchors between each pair.

    For example, with anchors [min, harmonic, geometric] and n_interp=1::

        min, (min+harmonic)/2, harmonic, (harmonic+geometric)/2, geometric

    Returns expanded (names, fns, andness_vals).
    """
    exp_names = [names[0]]
    exp_fns = [fns[0]]
    exp_andness = [andness_vals[0]]

    for i in range(len(names) - 1):
        for j in range(1, n_interp + 1):
            t = j / (n_interp + 1)
            interp_name = f"{names[i]}~{names[i+1]}@{t:.2f}"
            interp_fn = _make_interpolated_fn(fns[i], fns[i+1], t)
            interp_andness = (1.0 - t) * andness_vals[i] + t * andness_vals[i+1]
            exp_names.append(interp_name)
            exp_fns.append(interp_fn)
            exp_andness.append(interp_andness)
        exp_names.append(names[i+1])
        exp_fns.append(fns[i+1])
        exp_andness.append(andness_vals[i+1])

    return exp_names, exp_fns, exp_andness


class GenericGLAggregator(AggregatorBase, nn.Module):
    """Generic Graded Logic aggregator — N-ary canonical form.

    Implements the canonical GL aggregation (paper Section 4.5):

        A(x | c) = Σ αᵢ(ψ(u, c)) · Fᵢ(u),   u = R · x

    where Fᵢ are anchor operators from the generalized-mean family and
    the convex weights αᵢ are controlled by the Mean Andness Theorem.

    The six core anchors from the paper are::

        min  →  harmonic  →  geometric  →  mean  →  quadratic  →  max
        (α=1)   (α=3/4)     (α=5/8)      (α=1/2)  (α=3/8)      (α=0)

    Two additional t-norm / t-conorm anchors (``product``, ``prob_sum``)
    are available as extensions but are not part of the standard GL set.

    Parameters
    ----------
    anchors : sequence of str
        Anchor operators.  The paper's default set is
        ``('min', 'harmonic', 'geometric', 'mean', 'quadratic', 'max')``.
        Additional choices: ``'product'`` (algebraic t-norm, andness ≈ 1.08),
        ``'prob_sum'`` (probabilistic t-conorm, andness ≈ −0.08).
    anchor_interpolation : int
        Number of linearly interpolated anchors to insert between each
        consecutive pair.  For example, ``anchor_interpolation=1`` with the
        default 6 anchors produces 6 + 5 = 11 total anchors; the 5 extras
        sit at the midpoints of each adjacent pair.  ``0`` (default) adds
        no interpolated anchors.
    weight_mode : str
        * ``'static'``  — fixed learnable logits (MAT).
        * ``'conditional'``  — weights from context *c*.
        * ``'value_dependent'``  — weights from input features ψ(u).
        * ``'full'``  — weights from both inputs and context.
    use_transform : bool
        Learn a row-stochastic N×N coordinate transformation R.
    context_dim : int
        Dimension of external context *c*.
    hidden_dim : int
        Hidden width of the neural gating network.
    tau : float
        Softmax temperature.
    identity_reg : float
        Regularization toward identity R.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        anchors: Sequence[str] = ('min', 'harmonic', 'geometric',
                                   'mean', 'quadratic', 'max'),
        anchor_interpolation: int = 0,
        weight_mode: str = 'static',
        use_transform: bool = False,
        context_dim: int = 0,
        hidden_dim: int = 16,
        tau: float = 0.5,
        identity_reg: float = 0.0,
        eps: float = 1e-7,
    ):
        AggregatorBase.__init__(self)
        nn.Module.__init__(self)

        for name in anchors:
            if name not in ANCHOR_FUNCTIONS:
                raise ValueError(
                    f"Unknown anchor '{name}'. "
                    f"Choose from: {list(ANCHOR_FUNCTIONS.keys())}"
                )

        # Build the (possibly expanded) anchor list with interpolations
        base_names = list(anchors)
        base_fns = [ANCHOR_FUNCTIONS[n] for n in base_names]
        base_andness = [ANCHOR_ANDNESS.get(n, 0.5) for n in base_names]

        if anchor_interpolation > 0 and len(base_names) >= 2:
            names, fns, andness_vals = _expand_anchors_with_interpolation(
                base_names, base_fns, base_andness, anchor_interpolation
            )
        else:
            names, fns, andness_vals = base_names, base_fns, base_andness

        self._anchor_names = names
        self._anchor_fns = fns
        self._anchor_andness = andness_vals
        self._anchor_interpolation = anchor_interpolation
        self.k = len(names)

        self.weight_mode = weight_mode
        self.use_transform = use_transform
        self.context_dim = context_dim
        self._tau = max(tau, 1e-4)
        self._initial_tau = self._tau
        self.identity_reg = identity_reg
        self.eps = eps

        # Transform R is lazily sized on first use because N is unknown
        # until the tree calls aggregate().  Pre-allocate 2×2 for the
        # common 2-input case.
        if use_transform:
            self._init_transform(2)

        # ---- Weight computation ----
        if weight_mode == 'static':
            self.alpha_logits = nn.Parameter(torch.zeros(self.k))
        elif weight_mode in ('conditional', 'value_dependent', 'full'):
            feature_dim = 0
            if weight_mode in ('value_dependent', 'full'):
                feature_dim += _PSI_DIM
            if weight_mode in ('conditional', 'full'):
                feature_dim += max(context_dim, 1)
            self._feature_dim = feature_dim
            self.gate_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.k),
            )
        else:
            raise ValueError(
                f"Unknown weight_mode '{weight_mode}'. "
                "Choose from: 'static', 'conditional', 'value_dependent', 'full'."
            )

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _init_transform(self, n: int):
        """Allocate (or resize) the N×N transform logits."""
        logits = torch.zeros(n, n)
        logits.fill_diagonal_(3.0)  # near-identity init
        param = nn.Parameter(logits)
        # Keep a single registered name regardless of size
        if hasattr(self, 'r_logits'):
            delattr(self, 'r_logits')
        self.register_parameter('r_logits', param)

    def _ensure_transform(self, n: int, device: torch.device):
        """Ensure r_logits matches *n* and lives on *device*."""
        if not self.use_transform:
            return
        need_init = not hasattr(self, 'r_logits') or self.r_logits is None
        if not need_init and self.r_logits.size(0) == n:
            if self.r_logits.device != device:
                self.r_logits.data = self.r_logits.data.to(device)
            return
        self._init_transform(n)
        self.r_logits.data = self.r_logits.data.to(device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tau(self) -> float:
        return self._tau

    @tau.setter
    def tau(self, value: float):
        self._tau = max(value, 1e-4)

    def set_tau(self, new_tau: float):
        self._tau = max(new_tau, 1e-4)

    def anneal_tau(
        self,
        epoch: int,
        max_epochs: int,
        final_tau: float = 0.01,
        schedule: str = 'exponential',
    ) -> float:
        progress = min(epoch / max(max_epochs - 1, 1), 1.0)
        if schedule == 'exponential':
            ratio = final_tau / self._initial_tau
            new_tau = self._initial_tau * (ratio ** progress)
        elif schedule == 'linear':
            new_tau = self._initial_tau + (final_tau - self._initial_tau) * progress
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.set_tau(new_tau)
        return self._tau

    # ------------------------------------------------------------------
    # Coordinate transformation  (Section 4.2, generalised to N×N)
    # ------------------------------------------------------------------

    def get_transform_matrix(self) -> torch.Tensor:
        """Return the current N×N row-stochastic transformation matrix R."""
        if not self.use_transform:
            raise RuntimeError("Coordinate transformation is disabled")
        return F.softmax(self.r_logits, dim=1)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply u = R @ x.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[N, ...]`` where N is the number of inputs.

        Returns
        -------
        torch.Tensor
            Same shape — each row is a convex combination of inputs.
        """
        N = x.size(0)
        self._ensure_transform(N, x.device)
        R = self.get_transform_matrix()  # [N, N]
        shape = x.shape
        flat = x.reshape(N, -1)   # [N, B]
        u_flat = R @ flat          # [N, B]
        return u_flat.reshape(shape)

    def transform_regularization(self) -> torch.Tensor:
        """||R - I||^2  penalty."""
        if not self.use_transform or self.identity_reg == 0:
            return torch.tensor(0.0, device=self._param_device())
        R = self.get_transform_matrix()
        I = torch.eye(R.size(0), device=R.device)
        return self.identity_reg * ((R - I) ** 2).sum()

    # ------------------------------------------------------------------
    # Feature mapping psi  (Section 4.4.1, N-ary summary statistics)
    # ------------------------------------------------------------------

    @staticmethod
    def _psi(u: torch.Tensor) -> torch.Tensor:
        """Summary-statistic feature mapping for N inputs.

        Parameters
        ----------
        u : torch.Tensor
            Shape ``[N, ...]`` — transformed (or raw) inputs.

        Returns
        -------
        torch.Tensor
            Shape ``[..., 5]`` with ``[mean, min, max, std, product]``.
        """
        f_mean = u.mean(dim=0)
        f_min  = u.min(dim=0).values
        f_max  = u.max(dim=0).values
        f_std  = u.std(dim=0) if u.size(0) > 1 else torch.zeros_like(f_mean)
        f_prod = u.prod(dim=0)
        return torch.stack([f_mean, f_min, f_max, f_std, f_prod], dim=-1)

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        u: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute alpha_i.  Returns ``[k]`` or ``[k, ...]``."""
        if self.weight_mode == 'static':
            return F.softmax(self.alpha_logits / self._tau, dim=0)

        parts = []
        if self.weight_mode in ('value_dependent', 'full'):
            parts.append(self._psi(u))
        if self.weight_mode in ('conditional', 'full'):
            if c is None:
                raise ValueError(
                    f"weight_mode='{self.weight_mode}' requires context c"
                )
            if c.dim() == 0:
                c = c.unsqueeze(-1)
            parts.append(c)

        features = torch.cat(parts, dim=-1)
        logits = self.gate_net(features)
        weights = F.softmax(logits / self._tau, dim=-1)

        if weights.dim() == 1:
            return weights
        return weights.movedim(-1, 0)

    # ------------------------------------------------------------------
    # Anchor evaluation (N-ary)
    # ------------------------------------------------------------------

    def _compute_anchors(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate all anchors on u ``[N, ...]``.  Returns ``[k, ...]``."""
        return torch.stack([fn(u) for fn in self._anchor_fns], dim=0)

    # ------------------------------------------------------------------
    # Forward  (N-ary native)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute A(x | c).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[N, ...]`` — N input truth values in [0, 1].
        c : torch.Tensor, optional
            External context.

        Returns
        -------
        torch.Tensor
            Aggregated output ``[...]``, clamped to [0, 1].
        """
        u = self.transform(x) if self.use_transform else x
        ops = self._compute_anchors(u)       # [k, ...]
        w   = self._compute_weights(u, c)    # [k] or [k, ...]

        if self.weight_mode == 'static' and w.dim() < ops.dim():
            for _ in range(ops.dim() - w.dim()):
                w = w.unsqueeze(-1)

        out = (w * ops).sum(dim=0)
        return torch.clamp(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # AggregatorBase interface (used by BACON tree)
    # ------------------------------------------------------------------

    def aggregate_float(self, values: Sequence[float], a: float, weights: Sequence[float]) -> float:
        dev = self._param_device()
        x = torch.tensor(values, dtype=torch.float32, device=dev)
        return float(self.forward(x).item())

    def aggregate_tensor(
        self,
        values: Sequence[Any],
        a: Any,
        weights: Sequence[Any] | torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack(list(values), dim=0)  # [N, ...]
        return self.forward(x)

    def _param_device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device('cpu')

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_weights(
        self,
        u: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        with torch.no_grad():
            if self.weight_mode == 'static':
                w = self._compute_weights(torch.zeros(1))
            else:
                if u is None:
                    raise ValueError("u is required for non-static weight modes")
                w = self._compute_weights(u, c)
            return w.cpu().numpy()

    def effective_andness(
        self,
        u: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> float:
        w = self.get_weights(u, c)
        a = np.array(self._anchor_andness)
        return float(np.dot(w, a))

    def entropy(
        self,
        u: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> float:
        w = self.get_weights(u, c)
        return float(-np.sum(w * np.log(w + self.eps)))

    def entropy_loss(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        u = self.transform(x) if self.use_transform else x
        w = self._compute_weights(u, c)
        log_w = torch.log(w + self.eps)
        ent = -(w * log_w).sum(dim=0)
        return ent.mean() if ent.dim() > 0 else ent

    def describe(
        self,
        u: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> dict:
        w = self.get_weights(u, c)
        info = {
            'tau': self._tau,
            'anchors': self._anchor_names,
            'weights': w.tolist(),
            'dominant_op': self._anchor_names[int(np.argmax(w))],
            'dominant_weight': float(np.max(w)),
            'effective_andness': self.effective_andness(u, c),
            'entropy': self.entropy(u, c),
            'weight_mode': self.weight_mode,
            'use_transform': self.use_transform,
        }
        if self.use_transform and hasattr(self, 'r_logits') and self.r_logits is not None:
            info['R'] = self.get_transform_matrix().detach().cpu().numpy().tolist()
        return info

    def __repr__(self) -> str:
        return (
            f"GenericGLAggregator("
            f"anchors={self._anchor_names}, "
            f"weight_mode='{self.weight_mode}', "
            f"use_transform={self.use_transform}, "
            f"tau={self._tau:.4f})"
        )
