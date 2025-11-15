# mnist-addition-2/reasoners/ltn_reasoner.py

import torch
import torch.nn as nn


class LTNAdditionReasoner(nn.Module):
    """
    LTN-style reasoning for MNIST addition, operating ONLY on digit probabilities.

    Inputs:
      p1: [B,10] digit prob for first image
      p2: [B,10] digit prob for second image

    Outputs:
      sat: [B,19] satisfaction degree of each possible sum k in [0..18]
    """
    def __init__(self):
        super().__init__()

        # Precompute sum_idx[i,j] = i+j for i,j in [0..9]
        sum_idx = torch.arange(10).view(10, 1) + torch.arange(10).view(1, 10)
        # Register as buffer so it moves with the module across devices
        self.register_buffer("sum_idx", sum_idx)  # [10,10]

    def satisfaction_per_sum(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute sat[b,k] = Σ_{i+j=k} p1[b,i] * p2[b,j].

        Args:
          p1, p2: [B,10]

        Returns:
          sat: [B,19]
        """
        # T[b,i,j] = product t-norm for conjunction
        T = p1.unsqueeze(2) * p2.unsqueeze(1)      # [B,10,10]

        sats = []
        for k in range(19):
            mask_k = (self.sum_idx == k)          # [10,10] bool
            s_k = (T[:, mask_k]).sum(dim=1)       # [B]
            sats.append(s_k)
        sat = torch.stack(sats, dim=1)            # [B,19]
        return sat

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Forward = compute satisfaction per sum class.
        """
        return self.satisfaction_per_sum(p1, p2)

    @staticmethod
    def loss_from_sat(sat: torch.Tensor, y: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        LTN log-satisfaction loss:
          Loss = -mean log sat_y

        Args:
          sat: [B,19] satisfaction per sum
          y:   [B] integer labels in [0..18]
        """
        sat_y = sat.gather(1, y.view(-1, 1)).squeeze(1)  # [B]
        loss = -torch.log(sat_y + eps).mean()
        return loss

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: given p1, p2, y → compute sat and loss.
        """
        sat = self.satisfaction_per_sum(p1, p2)
        return self.loss_from_sat(sat, y)

    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Predict sum label by argmax over sat[b,k].
        """
        sat = self.satisfaction_per_sum(p1, p2)   # [B,19]
        y_hat = sat.argmax(dim=1)                 # [B]
        return y_hat

    @torch.no_grad()
    def print_pair_rules(self, loader, digit_model, device: torch.device):
        """
        Optional: rule-view similar to your old print_pair_rules, but purely
        using p1, p2 from the shared digit model.

        For each predicted sum k, accumulate contributions of digit pairs (i,j).
        """
        self.eval()
        digit_model.eval()

        pair_sum = torch.zeros(19, 10, 10, device=device)
        count_k  = torch.zeros(19, device=device)

        import torch.nn.functional as F

        with torch.no_grad():
            for (x1, x2), _ in loader:
                x1, x2 = x1.to(device), x2.to(device)

                logits1, feat1 = digit_model(x1)
                logits2, feat2 = digit_model(x2)
                p1 = F.softmax(logits1, dim=1)
                p2 = F.softmax(logits2, dim=1)

                sat = self.satisfaction_per_sum(p1, p2)    # [B,19]
                k = sat.argmax(1)                          # [B]

                outer = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]

                for kk in range(19):
                    mask = (k == kk).float().view(-1, 1, 1)
                    if mask.sum() == 0:
                        continue
                    pair_sum[kk] += (outer * mask).sum(dim=0)
                    count_k[kk]  += mask.sum()

        print("\n=== LTN Pairwise rules (valid digit pairs i∧j with i+j=k) ===")
        for kk in range(19):
            if count_k[kk] == 0:
                print(f"y_{kk}: (no samples)")
                continue
            contrib = pair_sum[kk] / count_k[kk]
            pairs = [(i, j, contrib[i, j].item()) for i in range(10) for j in range(10)]
            pairs.sort(key=lambda t: t[2], reverse=True)
            valid = [(i, j, v) for (i, j, v) in pairs if i + j == kk]
            human = " ∨ ".join([f"({i}∧{j})" for (i, j, _) in valid]) or "(none)"
            print(f"y_{kk}: {human}")
        print("=============================================================\n")
