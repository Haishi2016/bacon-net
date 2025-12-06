import torch
import torch.nn as nn
import itertools

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
    
    def initialize_from_coarse_permutation(self, coarse_perm, group_size=3, block_std=0.5):
        """
        Initialize the permutation matrix using a coarse-grained hard permutation.
        
        Args:
            coarse_perm: A permutation array for the coarse matrix (e.g., [2, 0, 1] for 3x3)
            group_size: How many rows/cols of the full matrix each coarse element represents
            block_std: Standard deviation for the normal distribution within each block
        """
        n = self.num_leaves
        k = len(coarse_perm)  # Size of coarse matrix
        
        # Initialize logits to large negative values (soft zeros)
        new_logits = torch.ones(n, n) * (-5.0)
        
        # For each coarse permutation mapping
        for coarse_row in range(k):
            coarse_col = coarse_perm[coarse_row]
            
            # Determine the block boundaries in the full matrix
            row_start = coarse_row * group_size
            row_end = min((coarse_row + 1) * group_size, n)
            col_start = coarse_col * group_size
            col_end = min((coarse_col + 1) * group_size, n)
            
            # Fill this block with normally distributed values (soft assignment within block)
            block_height = row_end - row_start
            block_width = col_end - col_start
            new_logits[row_start:row_end, col_start:col_end] = torch.randn(block_height, block_width) * block_std
        
        # Update the parameter
        with torch.no_grad():
            self.logits.copy_(new_logits)
    
    @staticmethod
    def generate_all_coarse_permutations(n, group_size=3):
        """
        Generate all possible permutations for a coarse-grained matrix.
        
        Args:
            n: Size of the full matrix
            group_size: How many elements per group
        
        Returns:
            List of all permutation arrays for the coarse matrix
        """
        k = (n + group_size - 1) // group_size  # Ceiling division
        return list(itertools.permutations(range(k)))

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
