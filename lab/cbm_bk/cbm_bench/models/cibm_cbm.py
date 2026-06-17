"""Concepts' Information Bottleneck CBM (CIBM).

Faithful re-implementation of the IB-regularised soft-joint CBM from
Galliamov et al., "Concepts' Information Bottleneck Models" (ICLR 2026,
arXiv:2602.14626), built on top of the same backbone / bottleneck as
``baseline_cbm`` so the two are directly comparable in this harness.

Architecture (paper App. D.1):

    image --[backbone CNN]--> features  z
          --[mu head]-------> mu(z)        (n_concepts)
          --[sigma head]----> log var(z)   (n_concepts)
          c_logits = mu + sigma * eps,  eps ~ N(0, 1)   (reparameterisation)
          --[sigmoid]-------> concept_probs
          --[task head]-----> class_logits (n_classes)

The concept layer is therefore *stochastic* q(c | z) = N(mu(z), diag(sigma(z)^2)).
Two information-bottleneck regularisers are supported (selected by ``variant``):

* ``"ibe"`` (L_E-CIB, eq. 6) -- *recommended*: add ``beta * I(X; C)`` where the
  mutual information is a Gaussian Monte-Carlo estimate over the batch.
* ``"ibb"`` (L_S-CIBM, eq. 4): add ``-(1 - beta) * H(C)`` (i.e. *maximise* the
  concept entropy ``H(C) = sum_i log sigma_i``). Crucially the entropy term's
  gradient is **stopped from flowing into the image backbone** (paper App. D.7);
  it updates only the sigma head.

The IB term is exposed via :meth:`ib_loss`, which the training engine adds to
``task_loss + concept_loss_weight * concept_loss``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models

from .base import CBMOutput
from ..registry import register_model


_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}

_LOG2PI = math.log(2.0 * math.pi)


def _make_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name not in _BACKBONES:
        raise KeyError(f"Unknown backbone '{name}'. Available: {sorted(_BACKBONES)}")
    ctor, weights = _BACKBONES[name]
    net = ctor(weights=weights if pretrained else None)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


class CIBM_CBM(nn.Module):
    """Variational soft-joint CBM with a concepts' information-bottleneck term."""

    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        variant: str = "ibe",
        beta: float = 0.5,
        mi_const: float = 1.0,
        beta_lr: float = -0.01,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
        concept_activation: str = "identity",
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in ("ibe", "ibb"):
            raise ValueError(f"variant must be 'ibe' or 'ibb', got {variant!r}")
        self.variant = variant
        # Activation applied to the sampled concept logits before the task head.
        # ``"identity"`` (faithful default) feeds the raw soft concept logits z
        # straight to a linear task head -- exactly the official StochasticMLP
        # (cls = Linear(z)) and the paper's App. D.1 (label predictor stacked on
        # the concept logits). ``"sigmoid"`` instead feeds bounded concept
        # probabilities. The IB MI/entropy terms always operate in logit
        # (mu/logvar) space, so they are unaffected by this choice.
        act = concept_activation.lower()
        if act not in ("sigmoid", "identity"):
            raise ValueError(
                f"concept_activation must be 'sigmoid' or 'identity', got {act!r}"
            )
        self.concept_activation = act
        self.beta = float(beta)
        # --- Constrained (dual-beta) IB, faithful to the official repo --------
        # The official CIBM optimises a CONSTRAINED objective, not a fixed
        # penalty: it targets a mutual-information level ``mi_const`` (nats) and
        # treats ``beta`` as a Lagrange multiplier updated by dual gradient
        # ascent each step -- ``beta += beta_lr * (mi_const - I(X;C))`` -- with
        # the IBE loss term ``beta * (mi_const - I(X;C))``. This self-regulates
        # I(X;C) toward ``mi_const`` instead of driving it monotonically to
        # zero. ``beta_lr == 0`` holds beta fixed (still the constrained form);
        # ``beta_lr < 0`` (e.g. -0.01) gives the paper's adaptive behaviour.
        # Only IBE uses the dual update; IBB keeps a fixed beta (the official
        # code does not adapt beta for the entropy surrogate).
        self.mi_const = float(mi_const)
        self.beta_lr = float(beta_lr)
        # Constraint (mi_const - MI) cached each forward for the dual update.
        self._last_constraint: float | None = None
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.backbone, feat_dim = _make_backbone(backbone, pretrained)
        # q(c | z): a mean head and a (log-variance) spread head, both 1-layer.
        self.mu_head = nn.Linear(feat_dim, n_concepts)
        self.logvar_head = nn.Linear(feat_dim, n_concepts)
        # q(y | c): linear label predictor reading the (soft) concept layer.
        self.task_head = nn.Linear(n_concepts, n_classes)

        # Cached per-forward state used by ``ib_loss`` (set during ``forward``).
        self._mu: torch.Tensor | None = None
        self._logvar: torch.Tensor | None = None
        self._c_sample: torch.Tensor | None = None
        self._eps: torch.Tensor | None = None
        self._feats: torch.Tensor | None = None
        # When True, concepts are *sampled* even in eval mode (the backbone
        # still runs deterministically). Used to measure concept-purity on the
        # stochastic channel rather than the deterministic mean.
        self.force_stochastic: bool = False

    def forward(self, x: torch.Tensor) -> CBMOutput:
        feats = self.backbone(x)
        mu = self.mu_head(feats)
        logvar = self.logvar_head(feats).clamp(self.logvar_min, self.logvar_max)
        sigma = torch.exp(0.5 * logvar)

        if self.training or self.force_stochastic:
            eps = torch.randn_like(mu)
            c_logits = mu + sigma * eps
        else:
            # Deterministic mean at eval time (standard for VIB-style models).
            eps = torch.zeros_like(mu)
            c_logits = mu

        # Cache for the IB regulariser.
        self._mu = mu
        self._logvar = logvar
        self._c_sample = c_logits
        self._eps = eps
        self._feats = feats

        if self.concept_activation == "sigmoid":
            concept_feats = torch.sigmoid(c_logits)
        else:
            concept_feats = c_logits
        class_logits = self.task_head(concept_feats)
        return CBMOutput(concept_logits=c_logits, class_logits=class_logits)

    # ------------------------------------------------------------------ #
    # Information-bottleneck regulariser
    # ------------------------------------------------------------------ #
    def ib_loss(self) -> torch.Tensor:
        """Signed IB term to *add* to the total loss for the current batch.

        For IBE returns the constrained term ``beta * (mi_const - I(X; C))``
        (with ``beta`` adapted by :meth:`ib_dual_step`); for IBB returns
        ``-(1 - beta) * H(C)``. Must be called immediately after :meth:`forward`
        (uses cached state).
        """
        if self._mu is None:
            raise RuntimeError("ib_loss() called before forward().")
        if not self.training:
            # No stochasticity at eval; the IB term is a no-op.
            return self._mu.new_zeros(())
        if self.variant == "ibe":
            # Constrained dual-beta IB (faithful to the official repo):
            #   loss term = beta * (mi_const - I(X;C))
            # beta is updated after the optimiser step by ``ib_dual_step`` using
            # the constraint cached here, regulating I(X;C) toward ``mi_const``.
            mi = self._mutual_information()
            constraint = self.mi_const - mi
            self._last_constraint = float(constraint.detach())
            return self.beta * constraint
        return -(1.0 - self.beta) * self._concept_entropy()

    def ib_dual_step(self) -> None:
        """Lagrangian dual update of ``beta`` (call once after each optim step).

        Faithful to the official CIBM: ``beta += beta_lr * (mi_const - I(X;C))``
        using the constraint cached by the most recent :meth:`ib_loss`. A no-op
        unless the constrained dual mode is active (``beta_lr != 0``, IBE). Beta
        is intentionally NOT clamped: the official update lets it cross through
        zero -- it settles at whatever (possibly negative) value holds I(X;C) at
        ``mi_const``, which is the entire point of the constrained formulation.
        """
        if self.beta_lr == 0.0 or self._last_constraint is None:
            return
        self.beta += self.beta_lr * self._last_constraint
        self._last_constraint = None

    def _concept_entropy(self) -> torch.Tensor:
        """H(C) = sum_i log sigma_i, averaged over the batch.

        The *summed* per-concept log-sigma (faithful to the official ``est_HC``
        and the paper, App. D.4). Gradient is stopped from reaching the
        backbone: the entropy is computed from log-variance predicted on
        *detached* features, so only the sigma head is updated by this term
        (paper App. D.7).
        """
        logvar_sg = self.logvar_head(self._feats.detach()).clamp(
            self.logvar_min, self.logvar_max
        )
        # log sigma_i = 0.5 * logvar_i ; drop additive constants.
        log_sigma = 0.5 * logvar_sg
        return log_sigma.sum(dim=1).mean()

    def _mutual_information(self) -> torch.Tensor:
        """Monte-Carlo estimate of I(X; C) for a diagonal-Gaussian q(c | z).

        I(X; C) = E[ log q(c | x) - log q(c) ], with the marginal q(c)
        approximated by the full empirical in-batch Gaussian mixture over all B
        components (including self), normalised by ``1/B``. Byte-identical to the
        official CIBM ``log_marg_prob`` (``-log(B) + logsumexp(logprob)``), so
        ``mi_const`` sits on the official nats scale. Returns the raw summed (over
        concepts) log-density MI the paper calibrates against (App. D.4).
        """
        mu = self._mu                      # (B, K)
        logvar = self._logvar              # (B, K)
        c = self._c_sample                 # (B, K)
        eps = self._eps                    # (B, K) ; c = mu + sigma * eps
        b, k = mu.shape

        # log q(c_b | x_b): Gaussian log-density at its own mean-shift.
        #   = sum_k [ -0.5 log(2 pi) - 0.5 logvar_bk - 0.5 eps_bk^2 ]
        log_q_c_given_x = (
            -0.5 * _LOG2PI * k
            - 0.5 * logvar.sum(dim=1)
            - 0.5 * (eps ** 2).sum(dim=1)
        )  # (B,)

        # log q(c_b) via the full in-batch mixture over all components b'.
        #   comp[b, b'] = sum_k logN(c_bk ; mu_b'k, sigma_b'k^2)
        c_e = c.unsqueeze(1)               # (B, 1, K)
        mu_e = mu.unsqueeze(0)             # (1, B, K)
        logvar_e = logvar.unsqueeze(0)     # (1, B, K)
        inv_var = torch.exp(-logvar_e)
        log_comp = -0.5 * (
            _LOG2PI + logvar_e + (c_e - mu_e) ** 2 * inv_var
        ).sum(dim=2)                       # (B, B)
        # Full mixture incl. self, normalised by 1/B -- exactly the official
        # ``-log(B) + logsumexp(logprob)``.
        log_q_c = torch.logsumexp(log_comp, dim=1) - math.log(b)  # (B,)

        return (log_q_c_given_x - log_q_c).mean()


@register_model("cibm_cbm")
def build_cibm_cbm(cfg, n_concepts: int, n_classes: int) -> CIBM_CBM:
    """Factory used by the training driver. ``cfg`` is the model config."""
    return CIBM_CBM(
        n_concepts=n_concepts,
        n_classes=n_classes,
        backbone=cfg.get("backbone", "resnet18"),
        pretrained=bool(cfg.get("pretrained", True)),
        variant=str(cfg.get("variant", "ibe")),
        beta=float(cfg.get("beta", 0.5)),
        mi_const=float(cfg.get("mi_const", 1.0)),
        beta_lr=float(cfg.get("beta_lr", -0.01)),
        concept_activation=str(cfg.get("concept_activation", "sigmoid")),
    )
