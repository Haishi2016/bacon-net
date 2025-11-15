# bacon/inputToCombinationSinkhorn.py

import torch
import torch.nn as nn


class inputToCombinationSinkhorn(nn.Module):
    """
    Differentiable combination layer + Sinkhorn:

    - Input:  x ∈ R^{B × num_inputs}, where num_inputs is typically 20.
              We split into two halves [0:split_index] and [split_index:].
    - Build all cross-half products (i from first half, j from second half),
      giving B × (split_index * (num_inputs - split_index)) "combination features".
    - Learn a doubly-stochastic matrix (via Sinkhorn, optionally with Gumbel noise)
      that maps these combination features to 'num_leaves' leaf nodes.

    This replaces the pure permutation layer with a combination-selection layer.
    """

    def __init__(
        self,
        num_inputs: int,
        num_leaves: int,
        split_index: int,
        temperature: float = 3.0,
        sinkhorn_iters: int = 20,
        use_gumbel: bool = True,
    ):
        super().__init__()
        assert 0 < split_index < num_inputs, \
            f"split_index must be in (0, {num_inputs}), got {split_index}"

        self.num_inputs = num_inputs
        self.num_leaves = num_leaves
        self.split_index = split_index
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.use_gumbel = use_gumbel
        self.gumbel_noise_scale = 1.0

        # Number of cross-half combinations
        self.num_pairs = split_index * (num_inputs - split_index)

        # Logits for Sinkhorn: each row is a leaf, each column is a pair (i,j)
        self.logits = nn.Parameter(torch.randn(num_leaves, self.num_pairs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_inputs]  (e.g., 20D: 10 for first digit, 10 for second digit)

        Steps:
          1. split into halves a, b
          2. build outer product a⊗b → [B, split, num_inputs-split]
          3. flatten to [B, num_pairs]
          4. apply Sinkhorn/Gumbel-Sinkhorn over self.logits to get P
          5. compute leaf_values = pair_vals @ P^T → [B, num_leaves]
        """
        B, D = x.shape
        assert D == self.num_inputs, f"Expected {self.num_inputs} inputs, got {D}"

        # 1) Split into two halves
        a = x[:, :self.split_index]            # [B, split]
        b = x[:, self.split_index:]            # [B, D - split]

        # 2) Outer product per sample: [B, split, D - split]
        outer = a.unsqueeze(2) * b.unsqueeze(1)

        # 3) Flatten 2D grid -> num_pairs
        pair_vals = outer.view(B, self.num_pairs)  # [B, num_pairs]

        # 4) Sinkhorn / Gumbel-Sinkhorn to get P ∈ R^{num_leaves × num_pairs}
        if self.use_gumbel:
            P = self._gumbel_sinkhorn(
                self.logits,
                temperature=self.temperature,
                n_iters=self.sinkhorn_iters,
                noise_scale=self.gumbel_noise_scale,
            )
        else:
            P = self._sinkhorn(
                self.logits,
                temperature=self.temperature,
                n_iters=self.sinkhorn_iters,
            )

        P = torch.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
        # 5) Map pair features to leaves
        # pair_vals: [B, num_pairs], P^T: [num_pairs, num_leaves]
        leaf_values = pair_vals @ P.t()        # [B, num_leaves]
        return leaf_values

    # ---- utilities (similar to inputToLeafSinkhorn) ----

    def _sample_gumbel(self, shape, device=None, eps=1e-20):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_sinkhorn(
        self,
        log_alpha: torch.Tensor,
        temperature: float = 1.0,
        n_iters: int = 20,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        noise = self._sample_gumbel(log_alpha.shape, device=log_alpha.device) * noise_scale
        perturbed = log_alpha + noise
        return self._sinkhorn(perturbed, temperature=temperature, n_iters=n_iters)

    def _sinkhorn(
        self,
        log_alpha: torch.Tensor,
        temperature: float = 1.0,
        n_iters: int = 20,
    ) -> torch.Tensor:
        log_alpha = log_alpha / temperature
        log_alpha = torch.clamp(log_alpha, min=-10, max=10)
        A = torch.exp(log_alpha)

        for _ in range(n_iters):
            A = A / A.sum(dim=1, keepdim=True)   # row-normalize
            A = A / A.sum(dim=0, keepdim=True)   # column-normalize

        return A
