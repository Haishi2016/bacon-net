"""Faithful port of dsb-ifi/cibm ``src/losses.py`` (the two terms used in training).

``est_MI``  -- I(X;C) = E[log q(c|x) - log q(c)], estimated over a random
              ``sz``-point subset of the dataset (the marginal q(c) is the
              ``sz``-component in-batch Gaussian mixture, self included).
``est_HC``  -- H(C) surrogate = sum of log(sigma) over the ``sz``-point subset
              (proportional to the Gaussian differential entropy).

Both sample ``sz`` points from the WHOLE dataset each call (not the current
minibatch), matching the official estimators.
"""

from __future__ import annotations

import numpy as np
import torch


def est_MI(model, dataset, sz, jensen, requires_grad: bool = True):
    """I(X;C) = H(C) - H(C|X) over a random ``sz``-point subset of ``dataset``."""
    device = next(model.parameters()).device
    ii = np.random.choice(len(dataset), size=sz, replace=False)
    x = torch.stack([torch.as_tensor(dataset[i][0]) for i in ii], dim=0).to(device)

    if not requires_grad:
        model.eval()
        with torch.no_grad():
            z, log_prob = model(x, return_log_prob=True)
            log_marg_prob = model.log_marg_prob(z, x, jensen=jensen)
        model.train()
    else:
        z, log_prob = model(x, return_log_prob=True)
        log_marg_prob = model.log_marg_prob(z, x, jensen=jensen)

    return (log_prob - log_marg_prob).mean()  # I(X;C) = H(C) - H(C|X)


def est_HC(model, dataset, sz, jensen: bool = False):
    """H(C) surrogate: sum of log(sigma) over a random ``sz``-point subset."""
    device = next(model.parameters()).device
    ii = np.random.choice(len(dataset), size=sz, replace=False)
    x = torch.stack([torch.as_tensor(dataset[i][0]) for i in ii], dim=0).to(device)
    enc = model.backbone(x) if model.backbone is not None else x
    sigma = model.pred_sigma(enc)
    # sum of log(sigma) is proportional to H(C) for Gaussian C.
    return torch.sum(torch.log(sigma + 1e-8))
