"""
Training policies for BACON models.

Policies allow injecting specific behaviors during training, such as:
- Fixed andness values
- Andness constraints (min/max bounds)
- Scheduled andness annealing
- Operator restrictions

Usage:
    from bacon.policies import FixedAndnessPolicy
    
    policy = FixedAndnessPolicy(andness=0.5)
    
    # During training loop, apply policy before forward pass:
    policy.apply(model)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, List, Union


class TrainingPolicy(ABC):
    """Base class for training policies."""
    
    def __init__(self, name: str = "base_policy"):
        self.name = name
        self.enabled = True
    
    @abstractmethod
    def apply(self, model: nn.Module) -> None:
        """Apply the policy to the model before forward pass."""
        pass
    
    def on_epoch_start(self, epoch: int, max_epochs: int) -> None:
        """Called at the start of each epoch. Override for scheduled policies."""
        pass
    
    def on_epoch_end(self, epoch: int, max_epochs: int) -> None:
        """Called at the end of each epoch. Override for adaptive policies."""
        pass
    
    def describe(self) -> dict:
        """Return a description of the policy."""
        return {"name": self.name, "enabled": self.enabled}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"


class FixedAndnessPolicy(TrainingPolicy):
    """
    Policy that encourages andness toward a target value via regularization.
    
    The andness parameter 'a' controls the aggregation behavior:
        a = 1.5:  Strong AND (product-like)
        a = 1.0:  Weak AND (min-like)
        a = 0.5:  Neutral (average)
        a = 0.0:  Weak OR (max-like)
        a = -0.5: Strong OR (prob-sum-like)
    
    This policy adds a penalty term to the loss that penalizes deviation from
    the target andness. The model can still learn to deviate if data supports it.
    
    Args:
        andness: Target andness value (default: 0.5 for neutral/average)
        penalty_weight: Weight of the regularization penalty (default: 1.0)
                       Higher values more strongly encourage the target andness.
        
    Example:
        # Encourage neutral andness with moderate penalty
        policy = FixedAndnessPolicy(andness=0.5, penalty_weight=1.0)
        
        # During training, add policy.penalty(model) to the loss
    """
    
    def __init__(self, andness: float = 0.5, penalty_weight: float = 1.0):
        super().__init__(name=f"andness_regularization_{andness}")
        self.andness = andness
        self.penalty_weight = penalty_weight
        
        # Compute the bias value that produces this andness
        # andness = sigmoid(bias) * 3 - 1
        # sigmoid(bias) = (andness + 1) / 3
        # bias = logit((andness + 1) / 3)
        sigmoid_target = (andness + 1) / 3.0
        # Clamp to avoid inf
        sigmoid_target = max(min(sigmoid_target, 0.9999), 0.0001)
        self.target_bias = torch.logit(torch.tensor(sigmoid_target)).item()
    
    def apply(self, model: nn.Module) -> None:
        """No-op for regularization policy - penalty is applied via loss."""
        pass
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the regularization penalty for bias deviation from target.
        
        Returns:
            Scalar tensor with the penalty value (to be added to loss).
        """
        if not self.enabled:
            return torch.tensor(0.0)
        
        # Access the tree's biases
        if hasattr(model, 'tree') and hasattr(model.tree, 'biases'):
            biases = model.tree.biases
        elif hasattr(model, 'biases'):
            biases = model.biases
        else:
            return torch.tensor(0.0)
        
        # Compute squared deviation from target bias
        # penalty = weight * sum((bias - target_bias)^2)
        penalty = torch.tensor(0.0, device=biases[0].device)
        for bias in biases:
            penalty = penalty + (bias - self.target_bias) ** 2
        
        return self.penalty_weight * penalty
    
    def release(self, model: nn.Module) -> None:
        """Disable the policy."""
        self.enabled = False
    
    def describe(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "andness": self.andness,
            "target_bias": self.target_bias,
            "penalty_weight": self.penalty_weight
        }


class AndnessRangePolicy(TrainingPolicy):
    """
    Policy that constrains andness to a specific range.
    
    Unlike FixedAndnessPolicy, this allows training but clamps
    the andness values to stay within bounds.
    
    Args:
        min_andness: Minimum andness value (default: 0.0)
        max_andness: Maximum andness value (default: 1.0)
    """
    
    def __init__(self, min_andness: float = 0.0, max_andness: float = 1.0):
        super().__init__(name=f"andness_range_{min_andness}_{max_andness}")
        self.min_andness = min_andness
        self.max_andness = max_andness
        
        # Compute bias bounds
        min_sigmoid = (min_andness + 1) / 3.0
        max_sigmoid = (max_andness + 1) / 3.0
        min_sigmoid = max(min(min_sigmoid, 0.9999), 0.0001)
        max_sigmoid = max(min(max_sigmoid, 0.9999), 0.0001)
        self.min_bias = torch.logit(torch.tensor(min_sigmoid)).item()
        self.max_bias = torch.logit(torch.tensor(max_sigmoid)).item()
    
    def apply(self, model: nn.Module) -> None:
        """Clamp biases to produce andness within range."""
        if not self.enabled:
            return
        
        if hasattr(model, 'tree') and hasattr(model.tree, 'biases'):
            biases = model.tree.biases
        elif hasattr(model, 'biases'):
            biases = model.biases
        else:
            return
        
        with torch.no_grad():
            for bias in biases:
                bias.clamp_(self.min_bias, self.max_bias)
    
    def describe(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "min_andness": self.min_andness,
            "max_andness": self.max_andness,
            "min_bias": self.min_bias,
            "max_bias": self.max_bias
        }


class ScheduledAndnessPolicy(TrainingPolicy):
    """
    Policy that anneals andness from one value to another over training.
    
    Useful for curriculum learning: start with simple operators (e.g., average)
    and gradually allow more complex ones.
    
    Args:
        start_andness: Initial andness value
        end_andness: Final andness value
        schedule: "linear" or "cosine"
    """
    
    def __init__(self, start_andness: float = 0.5, end_andness: float = 0.5,
                 schedule: str = "linear"):
        super().__init__(name=f"scheduled_andness_{start_andness}_to_{end_andness}")
        self.start_andness = start_andness
        self.end_andness = end_andness
        self.schedule = schedule
        self.current_andness = start_andness
        self._inner_policy = FixedAndnessPolicy(start_andness)
    
    def on_epoch_start(self, epoch: int, max_epochs: int) -> None:
        """Update andness based on training progress."""
        progress = min(epoch / max(max_epochs - 1, 1), 1.0)
        
        if self.schedule == "linear":
            self.current_andness = (
                self.start_andness + 
                (self.end_andness - self.start_andness) * progress
            )
        elif self.schedule == "cosine":
            import math
            self.current_andness = (
                self.end_andness + 
                (self.start_andness - self.end_andness) * 
                (1 + math.cos(math.pi * progress)) / 2
            )
        
        self._inner_policy = FixedAndnessPolicy(self.current_andness)
    
    def apply(self, model: nn.Module) -> None:
        """Apply the current scheduled andness."""
        if self.enabled:
            self._inner_policy.apply(model)
    
    def describe(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "start_andness": self.start_andness,
            "end_andness": self.end_andness,
            "current_andness": self.current_andness,
            "schedule": self.schedule
        }


class PolicyManager:
    """
    Manages multiple training policies.
    
    Example:
        manager = PolicyManager()
        manager.add(FixedAndnessPolicy(0.5))
        manager.add(SomeOtherPolicy())
        
        for epoch in range(max_epochs):
            manager.on_epoch_start(epoch, max_epochs)
            for batch in dataloader:
                manager.apply(model)  # Apply all policies
                output = model(batch)
                ...
            manager.on_epoch_end(epoch, max_epochs)
    """
    
    def __init__(self):
        self.policies: List[TrainingPolicy] = []
    
    def add(self, policy: TrainingPolicy) -> 'PolicyManager':
        """Add a policy to the manager."""
        self.policies.append(policy)
        return self
    
    def remove(self, policy_name: str) -> bool:
        """Remove a policy by name."""
        for i, p in enumerate(self.policies):
            if p.name == policy_name:
                self.policies.pop(i)
                return True
        return False
    
    def apply(self, model: nn.Module) -> None:
        """Apply all enabled policies to the model."""
        for policy in self.policies:
            if policy.enabled:
                policy.apply(model)
    
    def on_epoch_start(self, epoch: int, max_epochs: int) -> None:
        """Notify all policies of epoch start."""
        for policy in self.policies:
            policy.on_epoch_start(epoch, max_epochs)
    
    def on_epoch_end(self, epoch: int, max_epochs: int) -> None:
        """Notify all policies of epoch end."""
        for policy in self.policies:
            policy.on_epoch_end(epoch, max_epochs)
    
    def describe(self) -> List[dict]:
        """Return descriptions of all policies."""
        return [p.describe() for p in self.policies]
    
    def __len__(self) -> int:
        return len(self.policies)
    
    def __repr__(self) -> str:
        policy_strs = [repr(p) for p in self.policies]
        return f"PolicyManager([{', '.join(policy_strs)}])"
