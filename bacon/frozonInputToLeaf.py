import torch
import torch.nn as nn

class frozenInputToLeaf(nn.Module):
    def __init__(self, hard_assignment, num_inputs):
        super().__init__()
        self.register_buffer("P_hard", torch.zeros(len(hard_assignment), num_inputs))
        for leaf_idx, input_idx in enumerate(hard_assignment):
            self.P_hard[leaf_idx, input_idx] = 1.0

    def forward(self, x):
        return torch.matmul(x, self.P_hard.t().to(x.device))  # Ensure correct device
