import torch
import numpy as np
from typing import Callable, List, Tuple, Set


def add_uniform_noise(x: torch.Tensor, noise_ratio: float, seed: int = None) -> Tuple[torch.Tensor, List[Set[int]]]:
    """Add uniform noise to random features in each sample.
    
    Args:
        x: Input tensor of shape (batch_size, num_features)
        noise_ratio: Fraction of features to corrupt per sample (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (noisy_tensor, corrupted_indices_per_sample)
        where corrupted_indices_per_sample[i] is a set of feature indices corrupted in sample i
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    x_noisy = x.clone()
    batch_size, num_features = x.shape
    
    # Number of features to corrupt per sample
    num_corrupt = int(noise_ratio * num_features)
    
    corrupted_indices_per_sample = []
    
    if num_corrupt == 0:
        return x_noisy, [set() for _ in range(batch_size)]
    
    # For each sample, randomly select features to corrupt
    for i in range(batch_size):
        # Sample indices without replacement
        corrupt_indices = torch.randperm(num_features)[:num_corrupt]
        # Replace with uniform random values
        x_noisy[i, corrupt_indices] = torch.rand(num_corrupt, device=x.device)
        # Track which indices were corrupted
        corrupted_indices_per_sample.append(set(corrupt_indices.tolist()))
    
    return x_noisy, corrupted_indices_per_sample


def compute_nAUDC(
    accuracies: List[float], 
    noise_ratios: List[float]
) -> float:   
    if len(accuracies) != len(noise_ratios):
        raise ValueError("accuracies and noise_ratios must have same length")
    
    if 0.0 not in noise_ratios:
        raise ValueError("noise_ratios must include 0.0 (clean data)")
    
    # Sort by noise ratio
    sorted_pairs = sorted(zip(noise_ratios, accuracies))
    noise_ratios_sorted = [r for r, _ in sorted_pairs]
    accuracies_sorted = [a for _, a in sorted_pairs]
    
    # Get clean accuracy (at r=0)
    acc_clean = accuracies_sorted[0]
    
    if acc_clean == 0:
        return 0.0
    
    # Compute area under curve using trapezoidal rule
    auc = np.trapz(accuracies_sorted, noise_ratios_sorted)
    
    # Get max noise ratio
    r_max = max(noise_ratios)
    
    # Normalize
    nAUDC = auc / (r_max * acc_clean)
    
    return nAUDC
