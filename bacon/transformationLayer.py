"""
Transformation Layer for BACON Networks

This module provides a learnable transformation layer that can apply different
transformations to input features before aggregation. The transformations are
selected using a softmax mechanism, similar to the operator selection in aggregators.

The layer learns which features should be transformed (e.g., negated) to improve
the network's ability to learn complex logical relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformationLayer(nn.Module):
    """
    Learnable transformation layer that applies transformations to features.
    
    For each feature, the layer can select from a set of candidate transformations
    (e.g., identity x or negation 1-x) using a softmax-weighted combination.
    
    Args:
        num_features (int): Number of input features to transform.
        transformations (list[callable], optional): List of transformation functions.
            Each function should take a tensor and return a transformed tensor.
            Default: [identity, negation]
        temperature (float, optional): Temperature for softmax selection.
            Lower values make selection more discrete. Default: 1.0
        use_gumbel (bool, optional): Whether to use Gumbel-Softmax for sampling
            during training. Default: False
    
    Attributes:
        num_features (int): Number of features being transformed.
        num_transforms (int): Number of candidate transformations.
        logits (nn.Parameter): Learnable logits for transformation selection.
            Shape: (num_features, num_transforms)
        temperature (float): Softmax temperature.
        use_gumbel (bool): Whether to use Gumbel-Softmax.
    
    Example:
        >>> layer = TransformationLayer(num_features=10)
        >>> x = torch.rand(32, 10)  # batch_size=32, features=10
        >>> transformed = layer(x)
        >>> transformed.shape
        torch.Size([32, 10])
    """
    
    def __init__(
        self,
        num_features,
        transformations=None,
        temperature=1.0,
        use_gumbel=False
    ):
        super(TransformationLayer, self).__init__()
        
        self.num_features = num_features
        
        # Default transformations: identity and negation
        if transformations is None:
            self.transformations = [
                lambda x: x,           # Identity: x
                lambda x: 1.0 - x,     # Negation: 1-x
            ]
        else:
            self.transformations = transformations
        
        self.num_transforms = len(self.transformations)
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        
        # Learnable logits for selecting transformations
        # Shape: (num_features, num_transforms)
        # Each row represents the logits for selecting a transformation for that feature
        # Initialize with slight bias toward identity (first transformation)
        # This gives the model a stable starting point during permutation search
        initial_logits = torch.zeros(num_features, self.num_transforms)
        initial_logits[:, 0] = 0.5  # Slight bias toward identity transformation
        self.logits = nn.Parameter(initial_logits)
    
    def forward(self, x):
        """
        Apply transformations to input features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
        
        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, num_features).
        """
        batch_size = x.size(0)
        
        # Compute softmax weights for transformation selection
        # Shape: (num_features, num_transforms)
        if self.training and self.use_gumbel:
            # Use Gumbel-Softmax for differentiable discrete sampling
            weights = F.gumbel_softmax(
                self.logits,
                tau=self.temperature,
                hard=False,
                dim=-1
            )
        else:
            # Standard softmax
            weights = F.softmax(self.logits / self.temperature, dim=-1)
        
        # Apply each transformation and combine using softmax weights
        # For each feature, compute weighted sum of all transformations
        transformed = torch.zeros_like(x)
        
        for t_idx, transform_fn in enumerate(self.transformations):
            # Apply transformation to all features
            transformed_x = transform_fn(x)
            
            # Weight by softmax probabilities for this transformation
            # Broadcasting: weights[:, t_idx] is (num_features,)
            # transformed_x is (batch_size, num_features)
            # Result: each feature is weighted by its selection probability
            transformed += transformed_x * weights[:, t_idx].unsqueeze(0)
        
        return transformed
    
    def get_transformation_probabilities(self):
        """
        Get the current transformation selection probabilities for each feature.
        
        Returns:
            torch.Tensor: Probabilities of shape (num_features, num_transforms).
                Each row sums to 1.0 and represents the probability of selecting
                each transformation for that feature.
        """
        with torch.no_grad():
            probs = F.softmax(self.logits / self.temperature, dim=-1)
        return probs
    
    def get_selected_transformations(self):
        """
        Get the most likely transformation for each feature.
        
        Returns:
            torch.Tensor: Indices of shape (num_features,) indicating which
                transformation has the highest probability for each feature.
        """
        probs = self.get_transformation_probabilities()
        return torch.argmax(probs, dim=-1)
    
    def has_converged(self, confidence_threshold=0.8):
        """
        Check if the transformation layer has converged.
        
        A transformation layer is considered converged when each feature has
        selected a transformation with confidence above the threshold.
        
        Args:
            confidence_threshold (float): Minimum probability for a transformation
                to be considered "selected" (default: 0.8 = 80% confidence).
        
        Returns:
            bool: True if all features have selected transformations with high
                confidence, False otherwise.
        """
        probs = self.get_transformation_probabilities()
        max_probs = torch.max(probs, dim=-1)[0]  # Maximum probability for each feature
        return torch.all(max_probs >= confidence_threshold).item()
    
    def freeze_transformations(self):
        """
        Freeze the transformation selections to their current most likely values.
        This makes the layer deterministic and more interpretable.
        
        After freezing, each feature will use only its most probable transformation
        instead of a weighted combination.
        """
        selected = self.get_selected_transformations()
        
        # Create one-hot encoding of selected transformations
        # Shape: (num_features, num_transforms)
        one_hot = torch.zeros_like(self.logits)
        one_hot.scatter_(1, selected.unsqueeze(1), 1.0)
        
        # Set logits to produce this one-hot distribution
        # Use large values to make softmax nearly one-hot
        self.logits.data = one_hot * 10.0 - 5.0
        self.logits.requires_grad = False
    
    def unfreeze_transformations(self):
        """
        Unfreeze the transformation selections to allow further learning.
        """
        self.logits.requires_grad = True
    
    def get_transformation_summary(self):
        """
        Get a human-readable summary of transformation selections.
        
        Returns:
            dict: Dictionary mapping feature indices to their selected transformation
                and probability. Format:
                {
                    feature_idx: {
                        'transformation': 'identity' or 'negation',
                        'probability': float,
                        'all_probs': list of probabilities for all transformations
                    }
                }
        """
        probs = self.get_transformation_probabilities()
        selected = self.get_selected_transformations()
        
        transformation_names = ['identity', 'negation'] if self.num_transforms == 2 else [
            f'transform_{i}' for i in range(self.num_transforms)
        ]
        
        summary = {}
        for feat_idx in range(self.num_features):
            selected_idx = selected[feat_idx].item()
            summary[feat_idx] = {
                'transformation': transformation_names[selected_idx],
                'probability': probs[feat_idx, selected_idx].item(),
                'all_probs': probs[feat_idx].tolist()
            }
        
        return summary
    
    def __repr__(self):
        return (
            f"TransformationLayer(num_features={self.num_features}, "
            f"num_transforms={self.num_transforms}, "
            f"temperature={self.temperature}, "
            f"use_gumbel={self.use_gumbel})"
        )
