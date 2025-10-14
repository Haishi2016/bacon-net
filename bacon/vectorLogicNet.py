import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging
from scipy.optimize import linear_sum_assignment

from bacon.inputToLeafSinkhorn import inputToLeafSinkhorn
from bacon.frozonInputToLeaf import frozenInputToLeaf

class VectorLogicNet(nn.Module):
    """
    Left-associative graded-logic network parameterized by two vectors:
      - a:  (L,) andness per aggregation step (L = input_size - 1)
      - w:  (L,) weight for the NEW right input at each step; accumulator uses (1 - w)
    Keeps the same input-to-leaf (permutation) machinery as your original model.
    """
    def __init__(
        self,
        input_size,
        weight_mode="trainable",            # "trainable" | "fixed" | "range" | "discrete"
        weight_normalization="minmax",      # "minmax" | "softmax" (softmax becomes sigmoid for scalar) | "none"
        weight_value=0.5,
        weight_range=(0.0, 1.0),
        weight_choices=None,                # list/np.array of discrete choices for w (mapped via nearest)
        normalize_andness=True,
        loss_amplifier=1.0,
        weight_penalty_strength=1e-3,
        aggregator=None,                    # expects .aggregate(left, right, a, w_acc, w_new)
        is_frozen=False,
        lock_loss_tolerance=0.04,
        freeze_loss_threshold=0.07,
        permutation_max=10000,
        early_stop_patience=10,
        early_stop_min_delta=1e-4,
        early_stop_threshold=0.01,
        device=None
    ):
        super().__init__()
        self.original_input_size = input_size
        self.num_leaves = input_size
        self.L = input_size - 1

        self.weight_mode = weight_mode
        self.weight_normalization = weight_normalization
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.weight_choices = None
        self.normalize_andness = normalize_andness
        self.loss_amplifier = loss_amplifier
        self.weight_penalty_strength = weight_penalty_strength
        self.aggregator = aggregator

        self.is_frozen = is_frozen
        self.lock_loss_tolerance = lock_loss_tolerance * self.loss_amplifier
        self.freeze_loss_threshold = freeze_loss_threshold * self.loss_amplifier
        self.permutation_max = permutation_max

        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_threshold = early_stop_threshold

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weight_choices is not None:
            self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32, device=self.device)

        # Permutation module
        if not is_frozen:
            self.locked_perm = None
            self.input_to_leaf = inputToLeafSinkhorn(self.original_input_size, self.num_leaves, use_gumbel=True).to(self.device)
        else:
            best_perm = torch.arange(self.num_leaves, dtype=torch.long)
            self.locked_perm = best_perm.clone().detach()
            self.input_to_leaf = frozenInputToLeaf(best_perm, self.original_input_size).to(self.device)

        # Core parameters: vectors a and w (length L)
        # a_raw unconstrained; map with sigmoid * 3 - 1 if normalize_andness
        self.a_raw = nn.Parameter(torch.zeros(self.L))
        # w_raw unconstrained; map to (0,1) per normalization choice
        self.w_raw = nn.Parameter(torch.rand(self.L))  # start random in (0,1)

        # For fixed/range/discrete modes, we still store as parameters but clamp/replace in forward
        self.to(self.device)
        self.reset_optimizer()

    # ---------------- Utils ----------------
    def reset_optimizer(self, lr=0.2):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)

    def _map_andness(self):
        if self.normalize_andness:
            return torch.sigmoid(self.a_raw) * 3.0 - 1.0
        return self.a_raw

    def _map_weights(self):
        """
        Map w_raw -> w in (0,1) depending on mode/normalization.
        Returns w of shape (L,)
        """
        if self.weight_mode == "fixed":
            return torch.full_like(self.w_raw, float(self.weight_value))

        if self.weight_mode == "range":
            w = torch.sigmoid(self.w_raw)  # (0,1)
            lo, hi = self.weight_range
            return lo + (hi - lo) * w

        if self.weight_mode == "discrete":
            # project to nearest discrete choice
            if self.weight_choices is None or len(self.weight_choices) == 0:
                raise ValueError("weight_choices must be provided for 'discrete' mode.")
            cont = torch.sigmoid(self.w_raw)  # (0,1) as proxy
            choices = self.weight_choices.view(1, -1)  # (1,C)
            # Compute distances and pick nearest choice elementwise
            dists = torch.abs(cont.unsqueeze(-1) - choices)  # (L,C)
            idx = torch.argmin(dists, dim=-1)                # (L,)
            return self.weight_choices[idx]

        # trainable default
        if self.weight_normalization == "softmax":
            # For a scalar, softmax is degenerate; use sigmoid to map to (0,1)
            return torch.sigmoid(self.w_raw)
        elif self.weight_normalization == "minmax":
            # Min-max over the vector, then re-normalize to sum=1 across the 2 weights (acc & new)
            w = self.w_raw
            w_min = torch.min(w)
            w_max = torch.max(w)
            denom = (w_max - w_min).clamp_min(1e-8)
            w01 = (w - w_min) / denom                   # in [0,1]
            # Interpret as weight for NEW input; accumulator gets 1 - w
            return w01.clamp(0.0, 1.0)
        else:
            # "none" → clamp sigmoid
            return torch.sigmoid(self.w_raw)

    # -------------- Forward --------------
    def forward(self, x):
        """
        x: (batch, input_size)
        Steps:
          1) map inputs to leaves via current permutation module
          2) left-assoc fold: z0 = agg(x0, x1, a0, 1-w0, w0), then z1 = agg(z0, x2, a1, 1-w1, w1), ...
        returns: (batch, 1)
        """
        try:
            x = x.to(self.device)
            leaf_values = self.input_to_leaf(x)  # (batch, num_leaves)
            if torch.isnan(leaf_values).any():
                raise ValueError("NaNs detected in leaf_values")

            a_vec = self._map_andness()          # (L,)
            w_new = self._map_weights()          # (L,)
            w_acc = 1.0 - w_new                  # (L,)

            # Left-associative fold
            acc = leaf_values[:, 0]              # (batch,)
            right = leaf_values[:, 1]            # (batch,)
            z = self.aggregator.aggregate(acc, right, a_vec[0], w_acc[0], w_new[0])  # (batch,)

            for i in range(1, self.L):
                right = leaf_values[:, i + 1]
                z = self.aggregator.aggregate(z, right, a_vec[i], w_acc[i], w_new[i])

                if torch.isnan(z).any():
                    # If instability happens, replace NaNs with andness as a conservative fallback
                    z = torch.where(torch.isnan(z), a_vec[i].expand_as(z), z)

            return z.unsqueeze(1)  # (batch,1)
        except Exception as e:
            logging.error(f"[VectorLogicNet] forward error: {e}")
            logging.error(f"Input x: {x}")
            raise

    # -------------- Training w/ freeze (same behavior as your original) --------------
    def train_model(self, X, Y, epochs, is_frozen=False,
                    noise_increase=1.05, noise_decrease=0.95,
                    min_noise=0.0, max_noise=2.0):
        """
        Mirrors your training loop structure, incl. permutation freezing via sample_best_permutation.
        """
        # Re-init permutation module
        if not is_frozen:
            self.locked_perm = None
            self.input_to_leaf = inputToLeafSinkhorn(self.original_input_size, self.num_leaves, use_gumbel=True).to(self.device)
        else:
            best_perm = torch.arange(self.num_leaves, dtype=torch.long)
            self.locked_perm = best_perm.clone().detach()
            self.input_to_leaf = frozenInputToLeaf(best_perm, self.original_input_size).to(self.device)

        # noise controls (only meaningful for Sinkhorn variant with logits/noise)
        if hasattr(self.input_to_leaf, "gumbel_noise_scale"):
            self.input_to_leaf.gumbel_noise_scale = 1.0
        if hasattr(self.input_to_leaf, "temperature"):
            self.input_to_leaf.temperature = 1.0

        self.is_frozen = is_frozen
        self.reset_optimizer(lr=0.2)

        criterion = nn.BCELoss()
        loss_hist = []
        best_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            if hasattr(self.input_to_leaf, "temperature") and (epoch + 1) % 1000 == 0:
                self.input_to_leaf.temperature *= 0.8

            self.optimizer.zero_grad()
            out = self(X)  # (batch,1)
            if torch.isnan(out).any() or (out < 0).any() or (out > 1).any():
                raise RuntimeError("Instability detected. Can't be trained further.")

            loss = criterion(out, Y)

            # mild regularization pushing w toward 0.5 for interpretability balance
            if self.weight_penalty_strength > 0:
                w_new = self._map_weights()
                loss = loss + self.weight_penalty_strength * ((w_new - 0.5) ** 2).mean()

            loss = loss * self.loss_amplifier
            loss.backward()
            self.optimizer.step()

            # noise schedule (if available)
            if not self.is_frozen and hasattr(self.input_to_leaf, "gumbel_noise_scale"):
                loss_hist.append(loss.item())
                if len(loss_hist) > 6:
                    loss_hist.pop(0)
                    diffs = np.diff(loss_hist)
                    if all(d < 0 for d in diffs):
                        self.input_to_leaf.gumbel_noise_scale = max(self.input_to_leaf.gumbel_noise_scale * noise_decrease, min_noise)
                    elif all(abs(d) < 1e-4 for d in diffs):
                        self.input_to_leaf.gumbel_noise_scale = min(self.input_to_leaf.gumbel_noise_scale * noise_increase, max_noise)
                    elif any(d > 0 for d in diffs):
                        self.input_to_leaf.gumbel_noise_scale = min(self.input_to_leaf.gumbel_noise_scale * noise_increase, max_noise)

            # consider freezing permutation
            if not self.is_frozen and loss.item() < self.freeze_loss_threshold:
                best_model, best_perm, best_l, _ = self.sample_best_permutation(self, self.permutation_max, X, Y, noise_std=0.1)
                if best_model is not None and best_l < self.freeze_loss_threshold + self.lock_loss_tolerance:
                    # lock permutation
                    self.locked_perm = torch.tensor(best_perm, dtype=torch.long).clone().detach()
                    self.input_to_leaf = frozenInputToLeaf(self.locked_perm, self.original_input_size)
                    self.is_frozen = True
                    self.reset_optimizer(lr=0.02)
                    # reset early stopping counters for the frozen phase
                    best_loss = float('inf')
                    patience = 0
                    continue

            # early stopping once frozen
            if self.is_frozen:
                if loss.item() < best_loss - self.early_stop_min_delta:
                    best_loss = loss.item()
                    patience = 0
                else:
                    patience += 1

                if best_loss < self.early_stop_threshold:
                    logging.info(f"[VectorLogicNet] Early stop (low loss): {best_loss:.6f} @ epoch {epoch}")
                    break
                if patience >= self.early_stop_patience:
                    logging.info(f"[VectorLogicNet] Early stop (plateau). Best loss: {best_loss:.6f}")
                    break

            if epoch % 200 == 0:
                logging.info(f"[VectorLogicNet] epoch {epoch} loss {loss.item():.4f}")

        return

    # ---------- Sinkhorn helper & freezing (same pattern you used) ----------
    def _sinkhorn(self, log_alpha, n_iters=20, temperature=1.0):
        log_alpha = log_alpha / temperature
        A = torch.exp(log_alpha)
        for _ in range(n_iters):
            A = A / A.sum(dim=1, keepdim=True)
            A = A / A.sum(dim=0, keepdim=True)
        return A

    @torch.no_grad()
    def sample_best_permutation(self, model_template, topk, X, Y, noise_std=0.1):
        criterion = nn.BCELoss()
        if not hasattr(model_template.input_to_leaf, "logits"):
            raise ValueError("Permutation module has no learnable logits for Sinkhorn.")

        P = self._sinkhorn(model_template.input_to_leaf.logits,
                           temperature=getattr(model_template.input_to_leaf, "temperature", 1.0))
        P_np = P.cpu().numpy()

        perms = set()
        best_loss = float("inf")
        best_perm = None
        best_model = None
        best_index = None

        for index in range(topk * 2):
            if len(perms) == 0:
                noisy_P = P_np
            else:
                noise = np.random.normal(loc=0.0, scale=noise_std, size=P_np.shape)
                noisy_P = P_np + noise
            noisy_P = np.nan_to_num(noisy_P, nan=0.0, posinf=1e6, neginf=-1e6)
            row_ind, col_ind = linear_sum_assignment(-noisy_P)
            perm = tuple(col_ind[row_ind.argsort()])
            if perm in perms:
                continue
            perms.add(perm)

            perm_tensor = torch.tensor(perm, dtype=torch.long, device=self.device).clone().detach()
            temp_model = copy.deepcopy(model_template).to(self.device)
            temp_model.input_to_leaf = frozenInputToLeaf(perm_tensor, temp_model.original_input_size).to(self.device)
            X_dev = X.to(self.device)                                               # <-- add
            Y_dev = Y.to(self.device)                                               # <-- add
            temp_loss = criterion(temp_model(X_dev), Y_dev)            
            print(f"   🔍 Perm {perm} → Loss: {temp_loss.item():.4f}")

            if temp_loss > 0.9 and best_index is None:
                print(f"   🚫 Perm {perm} rejected (high loss).")
                return None, None, 1.0, -1

            if temp_loss < best_loss:
                best_loss = temp_loss
                best_model = temp_model
                best_perm = perm
                best_index = index

            if len(perms) >= topk:
                break

        if best_perm is None:
            return None, None, 1.0, -1

        print(f"✅ Best permutation selected: {best_perm} (Loss: {best_loss:.4f})")
        return best_model, best_perm, best_loss, best_index

    # ---------- Optional: prune like your original ----------
    def prune_features(self, features):
        """
        Prune first `features` inputs (requires frozen permutation).
        Returns a callable forward(x) -> (batch,1)
        """
        if not self.is_frozen:
            raise RuntimeError("Model is not frozen. Can't prune features.")
        if features >= self.num_leaves:
            raise ValueError(f"Cannot prune {features} >= {self.num_leaves}")

        # Effective remaining aggregation length
        Lr = self.L - features
        a_vec = self._map_andness()[features:]      # (Lr,)
        w_new = self._map_weights()[features:]      # (Lr,)
        w_acc = 1.0 - w_new

        def pruned_forward(x):
            leaf = x  # already permuted input is expected here; or call input_to_leaf if desired
            # If you want to keep permutation here, uncomment:
            # leaf = self.input_to_leaf(x)
            acc = leaf[:, 0]
            right = leaf[:, 1]
            z = self.aggregator.aggregate(acc, right, a_vec[0], w_acc[0], w_new[0])
            for i in range(1, Lr):
                right = leaf[:, i + 1]
                z = self.aggregator.aggregate(z, right, a_vec[i], w_acc[i], w_new[i])
            return z.unsqueeze(1)

        return pruned_forward
