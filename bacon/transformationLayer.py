"""
Transformation Layer for BACON Networks

This module provides a learnable transformation layer that can apply different
transformations to input features before aggregation. The transformations are
selected using a softmax mechanism, similar to the operator selection in aggregators.

The layer learns which features should be transformed (e.g., negated) to improve
the network's ability to learn complex logical relationships.

Supports parameterized transformations where each transformation can have
learnable parameters (e.g., peak location for Gaussian-like transformations).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterizedTransformation:
    """
    Base class for parameterized transformations.
    
    Each transformation can have its own learnable parameters that are
    optimized during training.
    
    Args:
        num_features (int): Number of features this transformation applies to.
        name (str): Human-readable name for this transformation.
    """
    def __init__(self, num_features, name):
        self.num_features = num_features
        self.name = name
    
    def initialize_parameters(self, device='cpu'):
        """
        Initialize learnable parameters for this transformation.
        
        Returns:
            dict: Dictionary of parameter names to nn.Parameter objects.
                  Empty dict if no parameters needed.
        """
        return {}
    
    def forward(self, x, params):
        """
        Apply the transformation to input features.
        
        Args:
            x (torch.Tensor): Input of shape (batch_size, num_features).
            params (dict): Dictionary of parameter tensors.
        
        Returns:
            torch.Tensor: Transformed input of same shape.
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_param_summary(self, params, feature_idx):
        """
        Get human-readable summary of parameters for a specific feature.
        
        Args:
            params (dict): Dictionary of parameter tensors.
            feature_idx (int): Index of the feature.
        
        Returns:
            dict: Summary of parameter values for this feature.
        """
        return {}


class IdentityTransformation(ParameterizedTransformation):
    """Identity transformation: f(x) = x"""
    def __init__(self, num_features):
        super().__init__(num_features, 'identity')
    
    def forward(self, x, params):
        return x


class NegationTransformation(ParameterizedTransformation):
    """Negation transformation: f(x) = 1 - x"""
    def __init__(self, num_features):
        super().__init__(num_features, 'negation')
    
    def forward(self, x, params):
        return 1.0 - x


class PeakTransformation(ParameterizedTransformation):
    """
    Peak transformation: High when x is near learned peak location t, low otherwise.
    
    Formula: f(x) = 1 - |x - t|
    
    This models features where an optimal value exists in the middle range,
    not at the extremes (e.g., age, income level).
    
    Each feature has its own learnable peak location t ∈ [0, 1].
    """
    def __init__(self, num_features):
        super().__init__(num_features, 'peak')
    
    def initialize_parameters(self, device='cpu'):
        # Initialize peak locations at 0.5 (middle of range)
        peak_locs = nn.Parameter(torch.ones(self.num_features, device=device) * 0.5)
        return {'peak_loc': peak_locs}
    
    def forward(self, x, params):
        # Get peak location for each feature
        # peak_loc shape: (num_features,)
        # x shape: (batch_size, num_features)
        # Apply sigmoid to keep peak_loc in [0,1] range
        peak_loc = torch.sigmoid(params['peak_loc'])
        
        # Compute distance from peak: |x - t|
        # Result shape: (batch_size, num_features)
        distance = torch.abs(x - peak_loc.unsqueeze(0))
        
        # Transform: 1 - |x - t|
        # High when near peak, low when far
        return 1.0 - distance
    
    def get_param_summary(self, params, feature_idx):
        # Note: Each transformation is initialized with num_features=1, so always use index 0
        peak_loc = torch.sigmoid(params['peak_loc'][0]).item()
        return {'peak_location': f"{peak_loc:.3f}"}


class ValleyTransformation(ParameterizedTransformation):
    """
    Valley (Negative Peak) transformation: Low when x is near learned valley location t, high otherwise.
    
    Formula: f(x) = |x - t|
    
    This models features where a specific value should be avoided (e.g., moderate debt is risky,
    very low or very high might be safer).
    
    Each feature has its own learnable valley location t ∈ [0, 1].
    """
    def __init__(self, num_features):
        super().__init__(num_features, 'valley')
    
    def initialize_parameters(self, device='cpu'):
        # Initialize valley locations at 0.5 (middle of range)
        valley_locs = nn.Parameter(torch.ones(self.num_features, device=device) * 0.5)
        return {'valley_loc': valley_locs}
    
    def forward(self, x, params):
        # Get valley location for each feature
        # Apply sigmoid to keep valley_loc in [0,1] range
        valley_loc = torch.sigmoid(params['valley_loc'])
        
        # Compute distance from valley: |x - t|
        distance = torch.abs(x - valley_loc.unsqueeze(0))
        
        # Transform: |x - t|
        # Low when near valley, high when far
        return distance
    
    def get_param_summary(self, params, feature_idx):
        # Note: Each transformation is initialized with num_features=1, so always use index 0
        valley_loc = torch.sigmoid(params['valley_loc'][0]).item()
        return {'valley_location': f"{valley_loc:.3f}"}


class StepUpTransformation(ParameterizedTransformation):
    """
    Step Up transformation: Ramps from 0 to 1 at threshold t, then stays at 1.
    
    Formula: f(x) = min(x/t, 1) for x ≥ 0, where division by t creates the ramp.
    More precisely: f(x) = 0 if x ≤ 0; (x/t) if 0 < x < t; 1 if x ≥ t
    
    This models features where values above a threshold are all equally important
    (e.g., "debt exists" vs "no debt").
    
    Each feature has its own learnable threshold t ∈ (0, 1].
    """
    def __init__(self, num_features):
        super().__init__(num_features, 'step_up')
    
    def initialize_parameters(self, device='cpu'):
        # Initialize thresholds at 0.5 (middle of range)
        # Add small epsilon to avoid division by zero
        thresholds = nn.Parameter(torch.ones(self.num_features, device=device) * 0.5)
        return {'threshold': thresholds}
    
    def forward(self, x, params):
        # Get threshold for each feature
        # Apply sigmoid to keep in [0,1], then scale to [0.01,1] to avoid division by zero
        threshold = torch.sigmoid(params['threshold']) * 0.99 + 0.01
        
        # Ramp: x/t, clamped to [0, 1]
        # When x < t: gradually increases from 0 to 1
        # When x ≥ t: stays at 1
        return torch.clamp(x / threshold.unsqueeze(0), min=0.0, max=1.0)
    
    def get_param_summary(self, params, feature_idx):
        # Each transformation instance has num_features=1, so params have shape (1,)
        threshold = (torch.sigmoid(params['threshold'][0]) * 0.99 + 0.01).item()
        return {'threshold': f"{threshold:.3f}"}


class StepDownTransformation(ParameterizedTransformation):
    """
    Step Down transformation: Ramps from 1 to 0 at threshold t, then stays at 0.
    
    Formula: f(x) = max(1 - x/t, 0) for x ≥ 0
    More precisely: f(x) = 1 if x ≤ 0; (1 - x/t) if 0 < x < t; 0 if x ≥ t
    
    This models features where values above a threshold are all equally unimportant
    (e.g., "low income is risky" but "medium and high income are equally safe").
    
    Each feature has its own learnable threshold t ∈ (0, 1].
    """
    def __init__(self, num_features):
        super().__init__(num_features, 'step_down')
    
    def initialize_parameters(self, device='cpu'):
        # Initialize thresholds at 0.5 (middle of range)
        thresholds = nn.Parameter(torch.ones(self.num_features, device=device) * 0.5)
        return {'threshold': thresholds}
    
    def forward(self, x, params):
        # Get threshold for each feature
        # Apply sigmoid to keep in [0,1], then scale to [0.01,1] to avoid division by zero
        threshold = torch.sigmoid(params['threshold']) * 0.99 + 0.01
        
        # Inverse ramp: 1 - x/t, clamped to [0, 1]
        # When x < t: gradually decreases from 1 to 0
        # When x ≥ t: stays at 0
        return torch.clamp(1.0 - x / threshold.unsqueeze(0), min=0.0, max=1.0)
    
    def get_param_summary(self, params, feature_idx):
        # Each transformation instance has num_features=1, so params have shape (1,)
        threshold = (torch.sigmoid(params['threshold'][0]) * 0.99 + 0.01).item()
        return {'threshold': f"{threshold:.3f}"}


class TransformationLayer(nn.Module):
    """
    Learnable transformation layer that applies transformations to features.
    
    For each feature, the layer can select from a set of candidate transformations
    (e.g., identity x, negation 1-x, peak) using a softmax-weighted combination.
    
    Supports parameterized transformations where each transformation can have
    learnable parameters (e.g., peak locations).
    
    Args:
        num_features (int): Number of input features to transform.
        transformations (list[ParameterizedTransformation], optional): List of transformation objects.
            Default: [IdentityTransformation, NegationTransformation]
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
        transform_params (nn.ParameterDict): Learnable parameters for each transformation.
    
    Example:
        >>> # Default: identity and negation
        >>> layer = TransformationLayer(num_features=10)
        >>> 
        >>> # With peak transformation
        >>> from bacon.transformationLayer import IdentityTransformation, NegationTransformation, PeakTransformation
        >>> transforms = [
        ...     IdentityTransformation(10),
        ...     NegationTransformation(10),
        ...     PeakTransformation(10)
        ... ]
        >>> layer = TransformationLayer(num_features=10, transformations=transforms)
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
        use_gumbel=False,
        device='cpu'
    ):
        super(TransformationLayer, self).__init__()
        
        self.num_features = num_features
        self.device = device
        
        # Default transformations: identity and negation
        if transformations is None:
            self.transformations = [
                IdentityTransformation(num_features),
                NegationTransformation(num_features)
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
        initial_logits = torch.zeros(num_features, self.num_transforms, device=device)
        initial_logits[:, 0] = 0.5  # Slight bias toward identity transformation
        self.logits = nn.Parameter(initial_logits)
        
        # Initialize parameters for each transformation
        # Use ParameterDict to properly register all parameters
        self.transform_params = nn.ParameterDict()
        for t_idx, transform in enumerate(self.transformations):
            params = transform.initialize_parameters(device=device)
            for param_name, param_tensor in params.items():
                # Store with unique key: "transform_{idx}_{param_name}"
                key = f"t{t_idx}_{param_name}"
                self.transform_params[key] = param_tensor
    
    def forward(self, x):
        """
        Apply learned transformations to input features.
        
        For each feature, selects from available transformations using softmax weights
        and computes a weighted combination of all transformation outputs.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
                Each feature value should be in [0, 1].
        
        Returns:
            torch.Tensor: Transformed features of shape (batch_size, num_features).
        
        Note:
            During training with use_gumbel=True, uses Gumbel-Softmax for discrete sampling.
            During evaluation, uses regular softmax for smooth interpolation.
        """
        batch_size = x.size(0)
        
        # Compute selection probabilities for each feature's transformation
        if self.training and self.use_gumbel:
            # Gumbel-Softmax for discrete sampling during training
            weights = F.gumbel_softmax(self.logits, tau=self.temperature, hard=False, dim=1)
        else:
            # Regular softmax for smooth interpolation
            weights = F.softmax(self.logits / self.temperature, dim=1)
        
        # weights shape: (num_features, num_transforms)
        
        # Apply each transformation and accumulate weighted results
        result = torch.zeros_like(x)
        
        for t_idx, transform in enumerate(self.transformations):
            # Get parameters for this transformation
            params = {}
            prefix = f"t{t_idx}_"
            for key, value in self.transform_params.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]  # Remove prefix
                    params[param_name] = value
            
            # Apply transformation
            # Shape: (batch_size, num_features)
            transformed = transform.forward(x, params)
            
            # Weight by selection probability for this transformation
            # weights[:, t_idx] has shape (num_features,)
            # Broadcast across batch: (batch_size, num_features)
            weighted_transformed = transformed * weights[:, t_idx].unsqueeze(0)
            
            # Accumulate
            result += weighted_transformed
        
        return result
    
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
    
    def has_converged(self, confidence_threshold=0.8, min_converged_ratio=0.8):
        """
        Check if the transformation layer has converged.
        
        A transformation layer is considered converged when a sufficient proportion
        of features have selected a transformation with confidence above the threshold.
        This is more flexible than requiring all features to converge, which can be
        too strict for larger feature sets where some features may not affect the output.
        
        Args:
            confidence_threshold (float): Minimum probability for a transformation
                to be considered "selected" (default: 0.8 = 80% confidence).
            min_converged_ratio (float): Minimum ratio of features that must have
                converged (default: 0.8 = 80% of features).
        
        Returns:
            bool: True if enough features have selected transformations with high
                confidence, False otherwise.
        """
        probs = self.get_transformation_probabilities()
        max_probs = torch.max(probs, dim=-1)[0]  # Maximum probability for each feature
        converged_features = (max_probs >= confidence_threshold).sum().item()
        total_features = len(max_probs)
        converged_ratio = converged_features / total_features
        return converged_ratio >= min_converged_ratio
    
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
                        'transformation': transformation name,
                        'probability': float,
                        'all_probs': list of probabilities for all transformations,
                        'params': dict of learned parameters (if any)
                    }
                }
        """
        probs = self.get_transformation_probabilities()
        selected = self.get_selected_transformations()
        
        # Get transformation names from the transformation objects
        transformation_names = []
        for transform in self.transformations:
            name = transform.__class__.__name__.replace('Transformation', '').lower()
            transformation_names.append(name)
        
        summary = {}
        for feat_idx in range(self.num_features):
            selected_idx = selected[feat_idx].item()
            
            # Get parameters for the selected transformation
            params = {}
            prefix = f"t{selected_idx}_"
            transform_params = {}
            for key, value in self.transform_params.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]  # Remove prefix
                    transform_params[param_name] = value
            
            # Get parameter summary from the transformation
            if transform_params:
                param_summary = self.transformations[selected_idx].get_param_summary(
                    transform_params, feat_idx
                )
            else:
                param_summary = {}
            
            summary[feat_idx] = {
                'transformation': transformation_names[selected_idx],
                'probability': probs[feat_idx, selected_idx].item(),
                'all_probs': probs[feat_idx].tolist(),
                'params': param_summary
            }
        
        return summary
    
    def __repr__(self):
        return (
            f"TransformationLayer(num_features={self.num_features}, "
            f"num_transforms={self.num_transforms}, "
            f"temperature={self.temperature}, "
            f"use_gumbel={self.use_gumbel})"
        )
