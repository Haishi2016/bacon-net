import numpy as np
import torch
import torch.nn as nn

# adapted from https://github.com/xu-ji/information-bottleneck
EPS = 1e-7


class StochasticMLP(nn.Module):
    def __init__(self, arch, activation='relu', backbone=None):
        super().__init__()
        self.backbone = backbone
        activation_class = nn.ReLU if activation == 'relu' else nn.Tanh
        self.arch = arch
        self.z_sz = arch[-2]

        layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - 2):
            layers += [activation_class(), nn.Linear(arch[l], arch[l + 1])]

        self.pred_mu = nn.Sequential(*layers)

        sigma_layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - 2):
            sigma_layers += [activation_class(), nn.Linear(arch[l], arch[l + 1])]

        self.pred_sigma = nn.Sequential(*(sigma_layers + [nn.Softplus()]))

        self.cls = nn.Sequential(
            nn.Linear(arch[-2], arch[-1]),
        )

    def get_activations(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        feats = [self.pred_mu(x)]
        return feats

    def forward(self, x, return_log_prob=False):
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)  # n, enc_sz
        stds = self.pred_sigma(x).clamp(min=EPS)

        eps = torch.randn_like(means)  # N(0, I)
        z = means + stds * eps

        if return_log_prob:
            distr = torch.distributions.normal.Normal(means, stds)  # batch_sz, L for each
            z_prob = z
            logprob = distr.log_prob(z_prob).sum(dim=1)  # batch_sz
            return z, logprob
        return self.cls(z), stds.mean().item(), z

    def uncertainties(self, x):
        """log p(c=0) under the predicted distribution; lower = more uncertain"""
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)
        stds = self.pred_sigma(x).clamp(min=EPS)
        distr = torch.distributions.normal.Normal(means, stds)
        return distr.log_prob(torch.zeros_like(means))

    def forward_tti(self, x, gt_concepts, allowed_idx):
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)  # n, enc_sz
        stds = self.pred_sigma(x).clamp(min=EPS)
        gt_concepts_logits = 2 * (gt_concepts - 0.5)  # {0,1} -> {-1,+1}
        eps = torch.randn_like(means)
        z = means + stds * eps
        # scale to match z's dynamic range
        gt_concepts_logits *= (torch.max(z) - torch.mean(z))
        z[:, allowed_idx] = gt_concepts_logits[:, allowed_idx]

        return self.cls(z)

    def log_marg_prob(self, z, d_x, jensen):
        if self.backbone is not None:
            d_x = self.backbone(d_x)
        batch_sz, L = z.shape
        batch_sz2 = d_x.shape[0]

        means = self.pred_mu(d_x)  # n, enc_sz
        stds = self.pred_sigma(d_x).clamp(min=EPS)

        # for each target, pass through each mean
        means = means.unsqueeze(0).expand(batch_sz, batch_sz2, L)
        stds = stds.unsqueeze(0).expand(batch_sz, batch_sz2, L)

        z = z.unsqueeze(1).expand(batch_sz, batch_sz2, L)

        distr = torch.distributions.normal.Normal(means, stds)
        z_prob = z
        logprob = distr.log_prob(z_prob)
        assert logprob.shape == (batch_sz, batch_sz2, L)

        logprob = logprob.sum(dim=2)  # (batch_sz, batch_sz2)
        if jensen:
            log_margprob = logprob.mean(dim=1)  # Jensen's upper bound
        else:
            log_margprob = - np.log(batch_sz2) + torch.logsumexp(logprob, dim=1)

        assert log_margprob.shape == (batch_sz,)

        return log_margprob  # batch_sz


class BasicMLP(nn.Module):

    def __init__(self, arch, activation='relu', backbone=None):
        super().__init__()
        self.arch = arch
        self.backbone = backbone
        activation_class = nn.ReLU if activation == 'relu' else nn.Tanh

        layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - 2):
            layers += [activation_class(), nn.Linear(arch[l], arch[l + 1])]
        self.enc = nn.Sequential(*layers)
        self.cls = nn.Sequential(
            activation_class(),
            nn.Linear(arch[-2], arch[-1])
        )

    def get_activations(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        feats = []
        for layer in self.enc[:-1]:
            x = layer(x)
            if 'ReLU' in str(layer) or 'Tanh' in str(layer):
                feats.append(torch.clone(x))  # just in case
        return feats

    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        h = self.enc(x)
        return self.cls(h), None, h

    def forward_tti(self, x, gt_concepts, allowed_idx):
        if self.backbone is not None:
            x = self.backbone(x)
        h = self.enc(x)
        gt_concepts_logits = (gt_concepts - 0.5) * 6  # {0,1} -> {-3,+3} to approximate sigmoid logits
        h[:, allowed_idx] = gt_concepts_logits[:, allowed_idx]

        return self.cls(h)
