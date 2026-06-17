"""Faithful port of dsb-ifi/cibm ``src/models.py``.

``StochasticMLP`` (CIBM, ``is_stochastic=True``) and ``BasicMLP`` (the
deterministic CBM baseline) reproduced verbatim. When training on cached
embeddings (the paper's default ``train_backbone=False``) ``backbone`` is
``None`` and ``x`` is the precomputed 2048-d InceptionV3 feature.

``arch`` is the full layer spec ``[in_dim, *hidden, num_concepts, num_classes]``
so ``arch[-2] == num_concepts`` is the concept (z) dimension and ``arch[-1]``
the number of classes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Matches the official ``EPS`` used to clamp the predicted std away from 0.
EPS = 1e-7


class StochasticMLP(nn.Module):
    """CIBM encoder q(c|x) = N(mu(x), sigma(x)) + linear label head q(y|c)."""

    def __init__(self, arch, activation: str = "relu", backbone=None):
        super().__init__()
        self.backbone = backbone
        activation_class = nn.ReLU if activation == "relu" else nn.Tanh
        self.arch = arch
        self.z_sz = arch[-2]

        layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - 2):
            layers += [activation_class(), nn.Linear(arch[l], arch[l + 1])]
        self.pred_mu = nn.Sequential(*layers)

        sigma_layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - 2):
            sigma_layers += [activation_class(), nn.Linear(arch[l], arch[l + 1])]
        # Softplus keeps sigma strictly positive (their parameterisation; NOT a
        # log-variance head).
        self.pred_sigma = nn.Sequential(*(sigma_layers + [nn.Softplus()]))

        # Label predictor reads the raw sampled concept logits z (no sigmoid).
        self.cls = nn.Sequential(nn.Linear(arch[-2], arch[-1]))

    def get_activations(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        return [self.pred_mu(x)]

    def forward(self, x, return_log_prob: bool = False):
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)                       # (n, num_concepts)
        stds = self.pred_sigma(x).clamp(min=EPS)      # (n, num_concepts)

        eps = torch.randn_like(means)                 # N(0, I)
        z = means + stds * eps                        # reparameterised sample

        if return_log_prob:
            distr = torch.distributions.normal.Normal(means, stds)
            logprob = distr.log_prob(z).sum(dim=1)    # log q(c|x), (n,)
            return z, logprob
        # (logits, mean std scalar for logging, concept sample z)
        return self.cls(z), stds.mean().item(), z

    def uncertainties(self, x):
        """log p(c=0) under the predicted distribution; lower = more uncertain."""
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)
        stds = self.pred_sigma(x).clamp(min=EPS)
        distr = torch.distributions.normal.Normal(means, stds)
        return distr.log_prob(torch.zeros_like(means))

    def forward_tti(self, x, gt_concepts, allowed_idx):
        """Test-time intervention: overwrite ``allowed_idx`` concepts with GT."""
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)
        stds = self.pred_sigma(x).clamp(min=EPS)
        gt_concepts_logits = 2 * (gt_concepts - 0.5)  # {0,1} -> {-1,+1}
        eps = torch.randn_like(means)
        z = means + stds * eps
        gt_concepts_logits *= (torch.max(z) - torch.mean(z))  # match z's range
        z[:, allowed_idx] = gt_concepts_logits[:, allowed_idx]
        return self.cls(z)

    def log_marg_prob(self, z, d_x, jensen: bool):
        """log q(c) via the empirical Gaussian mixture over the ``d_x`` batch.

        For each target ``z`` evaluates the Gaussian density under every
        component (each ``d_x`` row's mu/sigma) and combines them: the full
        mixture ``-log(B') + logsumexp_b' logprob`` (or Jensen's upper bound
        ``mean_b' logprob``). Includes the self term, normalised by 1/B'.
        """
        if self.backbone is not None:
            d_x = self.backbone(d_x)
        batch_sz, L = z.shape
        batch_sz2 = d_x.shape[0]

        means = self.pred_mu(d_x)
        stds = self.pred_sigma(d_x).clamp(min=EPS)

        means = means.unsqueeze(0).expand(batch_sz, batch_sz2, L)
        stds = stds.unsqueeze(0).expand(batch_sz, batch_sz2, L)
        z = z.unsqueeze(1).expand(batch_sz, batch_sz2, L)

        distr = torch.distributions.normal.Normal(means, stds)
        logprob = distr.log_prob(z)
        assert logprob.shape == (batch_sz, batch_sz2, L)

        logprob = logprob.sum(dim=2)                  # (batch_sz, batch_sz2)
        if jensen:
            log_margprob = logprob.mean(dim=1)        # Jensen's upper bound
        else:
            log_margprob = -np.log(batch_sz2) + torch.logsumexp(logprob, dim=1)

        assert log_margprob.shape == (batch_sz,)
        return log_margprob


class BasicMLP(nn.Module):
    """Deterministic CBM baseline (their ``BasicMLP``)."""

    def __init__(self, arch, activation: str = "relu", backbone=None):
        super().__init__()
        self.arch = arch
        self.backbone = backbone
        activation_class = nn.ReLU if activation == "relu" else nn.Tanh

        layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - 2):
            layers += [activation_class(), nn.Linear(arch[l], arch[l + 1])]
        self.enc = nn.Sequential(*layers)
        self.cls = nn.Sequential(activation_class(), nn.Linear(arch[-2], arch[-1]))

    def get_activations(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        feats = []
        for layer in self.enc[:-1]:
            x = layer(x)
            if "ReLU" in str(layer) or "Tanh" in str(layer):
                feats.append(torch.clone(x))
        return feats

    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        h = self.enc(x)
        # (logits, None [no stochastic std], concept activations h)
        return self.cls(h), None, h

    def forward_tti(self, x, gt_concepts, allowed_idx):
        if self.backbone is not None:
            x = self.backbone(x)
        h = self.enc(x)
        gt_concepts_logits = (gt_concepts - 0.5) * 6  # {0,1} -> {-3,+3}
        h[:, allowed_idx] = gt_concepts_logits[:, allowed_idx]
        return self.cls(h)
