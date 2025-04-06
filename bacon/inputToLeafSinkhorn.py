import torch
import torch.nn as nn

class inputToLeafSinkhorn(nn.Module):
    def __init__(self, num_inputs, num_leaves, temperature=3.0, sinkhorn_iters=20, use_gumbel=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_leaves = num_leaves
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.use_gumbel = use_gumbel
        self.gumbel_noise_scale = 1.0  # You can anneal this

        self.logits = nn.Parameter(torch.randn(num_leaves, num_inputs))

    def forward(self, x):
        if self.use_gumbel:
            P = self.gumbel_sinkhorn(self.logits, temperature=self.temperature, n_iters=self.sinkhorn_iters, noise_scale=self.gumbel_noise_scale)
        else:
            P = self.sinkhorn(self.logits, temperature=self.temperature, n_iters=self.sinkhorn_iters)
        P = torch.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.matmul(x, P.t())
   
    def sample_gumbel(self, shape, device=None, eps=1e-20):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_sinkhorn(self, log_alpha, temperature=1.0, n_iters=20, noise_scale=1.0):
        noise = self.sample_gumbel(log_alpha.shape, device=log_alpha.device) * noise_scale
        perturbed = log_alpha + noise
        return self.sinkhorn(perturbed, temperature=temperature, n_iters=n_iters)
    def sinkhorn(self, log_alpha, n_iters=20, temperature=1.0):
        log_alpha = log_alpha / temperature
        log_alpha = torch.clamp(log_alpha, min=-10, max=10) 
        A = torch.exp(log_alpha)

        for i in range(n_iters):
            A = A / A.sum(dim=1, keepdim=True)
            A = A / A.sum(dim=0, keepdim=True)

        return A
