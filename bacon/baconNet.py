import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from bacon.binaryTreeLogicNet import binaryTreeLogicNet
import logging
import os
from dataclasses import dataclass
from typing import Optional
from bacon.aggregators.base import AggregatorBase
from bacon.aggregators.lsp import FullWeightAggregator, HalfWeightAggregator, LspSoftmaxAggregator
from bacon.aggregators.lsp.generic_gl import GenericGLAggregator
from bacon.aggregators.bool import MinMaxAggregator
from bacon.aggregators.math import OperatorSetAggregator, BoolOperatorSet, BoolOperatorSetWithIdentity, ArithmeticOperatorSet

_aggregator_registry = {
    "lsp.full_weight": FullWeightAggregator,
    "lsp.half_weight": HalfWeightAggregator,
    "lsp.softmax": LspSoftmaxAggregator,
    "gl.generic": GenericGLAggregator,
    "bool.min_max": MinMaxAggregator,
    "math.operator_set.logic": BoolOperatorSet,
    "math.operator_set.logic_identity": BoolOperatorSetWithIdentity,
    "math.operator_set.arith": ArithmeticOperatorSet,
}

@dataclass
class TrainingSetup:
    """Encapsulates all training setup state for a single attempt."""
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    pos_weight: Optional[float]
    param_groups: list
    aggregation_frozen: bool
    actual_max_epochs: int
    use_temperature_annealing: bool
    perm_temp_decay_rate: Optional[float]
    trans_temp_decay_rate: Optional[float]
    operator_tau_decay_rate: Optional[float]  # For operator selection annealing
    operator_initial_tau: Optional[float]  # Starting tau for operator selection
    anneal_over_epochs: int
    original_sparsity_weight: Optional[float]
    task_type: str  # "classification" or "regression"
    # Tracking variables
    loss_history: list
    accuracy_history: list
    freeze_confidence_warning_shown: bool
    best_loss_for_convergence: float
    epochs_without_improvement: int
    epoch_when_frozen: Optional[int]
    has_converged_before_freeze: bool
    best_confidence: float
    epochs_without_confidence_improvement: int
    best_loss: float
    epochs_since_improvement: int
    temp_paused: bool
    transformation_converged: bool
    # Frozen training stability tracking
    best_frozen_loss: Optional[float]
    best_frozen_state: Optional[dict]
    frozen_lr_reduced: bool
    # Best overall state tracking (across entire training)
    best_overall_loss: Optional[float]
    best_overall_state: Optional[dict]
    best_overall_epoch: Optional[int]
class baconNet(nn.Module):
    """
    Represents a BACON network for interpretable decision-making using graded logic.

    Args:
        input_size (int): Number of input features. This is likely to be removed in the future.
        tree_layout (str, optional): Layout of the tree. Defaults to "left". Other layouts are not supported yet.
        loss_amplifier (float, optional): Amplifier for the loss. Defaults to 1.
        weight_penalty_strength (float, optional): Penalty strength on weights. Defaults to 1e-3. A strong penalty leads to more balaned weights (closer to 0.5).
        normalize_andness (bool, optional): Whether to normalize andness. Defaults to True. This should set to False if the chosen aggregator, such as `bool.min_max`, already normalizes the andness.
        weight_mode (str, optional): Mode for weight configuration. Defaults to "trainable". Use "fixed" for fixed weights (set to 0.5).
        aggregator (str, optional): Aggregator to be used. Defaults to "lsp.full_weight".
        is_frozen (bool, optional): Whether to freeze the structure. Defaults to False.
        early_stop_threshold_large_inputs (float, optional): Early stop threshold for transformation layers with 10+ inputs. Defaults to 0.1. Lower values require more training but achieve higher accuracy.
        transformations (list, optional): List of transformation objects to use. If None, uses all 6 default transformations.
                                          Example: [IdentityTransformation(n), NegationTransformation(n)] for identity+negation only.
        permutation_initial_temperature (float, optional): Starting temperature for permutation annealing. Defaults to 5.0. Higher = more initial exploration.
        permutation_final_temperature (float, optional): Final temperature for permutation annealing. Defaults to 0.1. Lower = harder final permutation.
        transformation_initial_temperature (float, optional): Starting temperature for transformation layer. Defaults to 1.0. Should be lower than permutation since transformation is simpler (2^n vs n! states).
        transformation_final_temperature (float, optional): Final temperature for transformation layer. Defaults to 0.1. Same as permutation final temp.
        loss_weight_main (float, optional): Weight for main BCE loss. Defaults to 1.0. 
        loss_weight_perm_entropy (float, optional): Weight for permutation entropy regularization. Defaults to 0.0. Higher = encourage exploration. Typical range: 0.0-0.1.
        loss_weight_trans_entropy (float, optional): Weight for transformation entropy regularization. Defaults to 0.0. Higher = encourage decisive transformation selection. Typical range: 0.0-0.1.
        loss_weight_perm_sparsity (float, optional): Weight for permutation sparsity loss. Defaults to 0.01. Penalizes high entropy (flat distributions) to encourage peaked/sparse permutations. Higher = stronger push toward confident or clear multi-modal distributions. Typical range: 0.0-0.1.
        loss_weight_operator_sparsity (float, optional): Weight for operator selection sparsity loss. Defaults to 1.0. Penalizes uncertain operator choices to encourage commitment. Higher = faster operator decision.
        loss_weight_operator_l2 (float, optional): Weight for L2 regularization on operator logits. Defaults to 0.0. Keeps logits bounded so tau can control commitment timing. Use > 0 when operators commit too early.
        lr_permutation (float, optional): Learning rate for permutation layer. Defaults to 0.3. Higher = faster exploration of feature orderings.
        lr_transformation (float, optional): Learning rate for transformation layer. Defaults to 0.5. Higher = faster transformation selection.
        lr_aggregator (float, optional): Learning rate for aggregator weights. Defaults to 0.1. Lower = more stable tree structure.
        lr_other (float, optional): Learning rate for other parameters. Defaults to 0.1.
        use_class_weighting (bool, optional): Whether to apply class weighting for imbalanced data. Defaults to True. When True, penalizes minority class errors more heavily (pos_weight = neg_count/pos_count). When False, uses standard BCE loss (original behavior).
        full_tree_depth (int, optional): Depth of the fully connected tree. Only used when tree_layout="full". Defaults to None (uses input_size - 1).
        full_tree_shape (str, optional): Shape of the fully connected tree. "triangle" (default) or "square".
        full_tree_temperature (float, optional): Initial temperature for sigmoid edge weights. Defaults to 3.0.
        full_tree_final_temperature (float, optional): Final temperature after annealing. Defaults to 0.1.
        full_tree_max_egress (int, optional): Each source concentrates on top-K destinations (via loss). Defaults to None (no constraint).
        loss_weight_full_tree_egress (float, optional): Weight for full tree egress constraint loss. Defaults to 0.0.
        loss_weight_full_tree_ingress (float, optional): Weight for full tree ingress constraint loss (max 2 inputs per node). Defaults to 0.5.
        use_permutation_layer (bool, optional): Whether to use the permutation layer. Defaults to True.
            Set to False for full tree layout to let the tree learn input routing directly.
        regression_loss_type (str, optional): Loss type for regression mode. Defaults to "mse".
            Supported values are "mse" (standard MSE, scale-sensitive),
            "correlation" (Pearson correlation loss, scale-invariant), and
            "normalized_mse" (z-score normalized MSE for scale-invariant
            pattern matching).
    """
    def __init__(self, input_size, 
                 tree_layout="left", 
                 loss_amplifier=1, 
                 weight_penalty_strength=1e-3,
                 weight_mode="trainable",
                 weight_normalization="minmax",
                 aggregator="lsp.full_weight",
                 normalize_andness=True,
                 is_frozen=False,
                 use_transformation_layer=False,
                 transformation_temperature=None,
                 transformation_use_gumbel=False,
                 transformations=None,
                 early_stop_threshold_large_inputs=0.1,
                 permutation_initial_temperature=5.0,
                 permutation_final_temperature=0.1,
                 transformation_initial_temperature=1.0,
                 transformation_final_temperature=0.1,
                 loss_weight_main=1.0,
                 loss_weight_perm_entropy=0.0,
                 loss_weight_trans_entropy=0.0,
                 loss_weight_perm_sparsity=0.01,
                 loss_weight_operator_sparsity=1.0,
                 loss_weight_operator_l2=0.0,
                 lr_permutation=0.3,
                 lr_transformation=0.5,
                 lr_aggregator=0.1,
                 lr_other=0.1,
                 use_class_weighting=True,
                 loss_trim_percentile: float = 0.0,
                 loss_trim_mode: str = "drop_high",
                 loss_trim_start_epoch: int = 0,
                 training_policy=None,
                 # Full tree parameters
                 full_tree_depth: int = None,
                 full_tree_shape: str = "triangle",
                 full_tree_temperature: float = 3.0,
                 full_tree_final_temperature: float = 0.1,
                 full_tree_max_egress: int = None,
                 full_tree_concentrate_ingress: bool = False,
                 full_tree_use_sinkhorn: bool = False,
                 loss_weight_full_tree_egress: float = 0.0,
                 loss_weight_full_tree_ingress: float = 0.5,
                 loss_weight_full_tree_ingress_balance: float = 0.0,
                 loss_weight_full_tree_scale_reg: float = 0.0,
                 # Alternating tree parameters
                 alternating_learn_first_routing: bool = True,
                 alternating_learn_subsequent_routing: bool = True,
                 alternating_learn_exponents: bool = False,
                 alternating_min_exponent: float = 1.0,
                 alternating_max_exponent: float = 2.0,
                 alternating_max_egress: int = 1,
                 alternating_use_straight_through: bool = True,
                 loss_weight_alternating_balance: float = 50.0,
                 loss_weight_alternating_egress: float = 0.5,
                 loss_weight_alternating_exponent_reg: float = 0.0,
                 use_constant_input: bool = False,
                 use_permutation_layer: bool = True,
                 regression_loss_type: str = "mse"):
        super(baconNet, self).__init__()        
        if isinstance(aggregator, str):
            if aggregator not in _aggregator_registry:
                raise ValueError(f"Unknown aggregator: {aggregator}. Available options: {list(_aggregator_registry.keys())}")
            aggregator_class = _aggregator_registry[aggregator]
            aggregator = aggregator_class()
        elif not isinstance(aggregator, AggregatorBase):
            raise TypeError(f"aggregator must be a string name or an AggregatorBase instance, got {type(aggregator).__name__}")
        # Keep a pristine copy for re-creating fresh aggregators across attempts
        self._original_aggregator = copy.deepcopy(aggregator)
        self.is_frozen = is_frozen
        self.early_stop_threshold_large_inputs = early_stop_threshold_large_inputs
        self.permutation_initial_temperature = permutation_initial_temperature
        self.permutation_final_temperature = permutation_final_temperature
        self.transformation_initial_temperature = transformation_initial_temperature
        self.transformation_final_temperature = transformation_final_temperature
        
        # Training policy (e.g., FixedAndnessPolicy)
        self.training_policy = training_policy
        
        # Use transformation_initial_temperature if transformation_temperature not specified
        if transformation_temperature is None:
            transformation_temperature = transformation_initial_temperature
        
        # Loss component weights (normalized internally during training)
        self.loss_weight_main = loss_weight_main  # Main BCE loss
        self.loss_weight_perm_entropy = loss_weight_perm_entropy  # Permutation entropy regularization
        self.loss_weight_trans_entropy = loss_weight_trans_entropy  # Transformation entropy regularization
        self.loss_weight_perm_sparsity = loss_weight_perm_sparsity  # Permutation sparsity loss (encourage peaked distributions)
        self.loss_weight_operator_sparsity = loss_weight_operator_sparsity  # Operator selection sparsity (encourage commitment)
        self.loss_weight_operator_l2 = loss_weight_operator_l2  # L2 on operator logits (prevent premature commitment)
        
        # Full tree loss weights
        self.loss_weight_full_tree_egress = loss_weight_full_tree_egress
        self.loss_weight_full_tree_ingress = loss_weight_full_tree_ingress
        self.loss_weight_full_tree_ingress_balance = loss_weight_full_tree_ingress_balance
        self.loss_weight_full_tree_scale_reg = loss_weight_full_tree_scale_reg
        
        # Alternating tree loss weights
        self.alternating_learn_first_routing = alternating_learn_first_routing
        self.alternating_learn_subsequent_routing = alternating_learn_subsequent_routing
        self.alternating_learn_exponents = alternating_learn_exponents
        self.alternating_min_exponent = alternating_min_exponent
        self.alternating_max_exponent = alternating_max_exponent
        self.alternating_max_egress = alternating_max_egress
        self.alternating_use_straight_through = alternating_use_straight_through
        self.loss_weight_alternating_balance = loss_weight_alternating_balance
        self.loss_weight_alternating_egress = loss_weight_alternating_egress
        self.loss_weight_alternating_exponent_reg = loss_weight_alternating_exponent_reg
        self.use_constant_input = use_constant_input
        
        # Full tree parameters (stored for reference)
        self.full_tree_depth = full_tree_depth
        self.full_tree_shape = full_tree_shape
        self.full_tree_temperature = full_tree_temperature
        self.full_tree_final_temperature = full_tree_final_temperature
        self.full_tree_max_egress = full_tree_max_egress
        self.full_tree_concentrate_ingress = full_tree_concentrate_ingress
        self.full_tree_use_sinkhorn = full_tree_use_sinkhorn
        self.use_permutation_layer = use_permutation_layer
        
        # Regression loss type: "mse" or "correlation"
        self.regression_loss_type = regression_loss_type
        
        # Learning rates for different parameter groups
        self.lr_permutation = lr_permutation
        self.lr_transformation = lr_transformation
        self.lr_aggregator = lr_aggregator
        self.lr_other = lr_other
        
        # Class weighting for imbalanced data
        self.use_class_weighting = use_class_weighting
        # Loss trimming hyperparameters
        if loss_trim_percentile < 0.0 or loss_trim_percentile >= 1.0:
            raise ValueError("loss_trim_percentile must be in [0.0, 1.0).")
        if loss_trim_mode not in ("drop_high", "drop_low", "none"):
            raise ValueError("loss_trim_mode must be 'drop_high', 'drop_low', or 'none'.")
        self.loss_trim_percentile = float(loss_trim_percentile)
        self.loss_trim_mode = loss_trim_mode
        if loss_trim_start_epoch < 0:
            raise ValueError("loss_trim_start_epoch must be >= 0.")
        self.loss_trim_start_epoch = int(loss_trim_start_epoch)
        
        import logging
        logging.info(f"🔧 Creating baconNet with transformations: {transformations}")
        if transformations:
            logging.info(f"   Number of custom transformations: {len(transformations)}")
            logging.info(f"   Types: {[type(t).__name__ for t in transformations]}")
        
        self.assembler = binaryTreeLogicNet(input_size, 
                                            weight_mode=weight_mode,
                                            weight_value=0.5,                                             
                                            weight_range=(0.5, 2.0), 
                                            normalize_andness=normalize_andness,
                                            tree_layout=tree_layout,
                                            loss_amplifier=loss_amplifier,
                                            is_frozen = is_frozen,
                                            weight_normalization=weight_normalization,
                                            aggregator=aggregator,
                                            weight_penalty_strength = weight_penalty_strength,
                                            use_transformation_layer=use_transformation_layer,
                                            transformation_temperature=transformation_temperature,
                                            transformation_use_gumbel=transformation_use_gumbel,
                                            transformations=transformations,
                                            weight_choices=None,
                                            # Full tree parameters
                                            full_tree_depth=full_tree_depth,
                                            full_tree_shape=full_tree_shape,
                                            full_tree_temperature=full_tree_temperature,
                                            full_tree_final_temperature=full_tree_final_temperature,
                                            full_tree_max_egress=full_tree_max_egress,
                                            full_tree_concentrate_ingress=full_tree_concentrate_ingress,
                                            full_tree_use_sinkhorn=full_tree_use_sinkhorn,
                                            # Alternating tree parameters
                                            alternating_learn_first_routing=alternating_learn_first_routing,
                                            alternating_learn_subsequent_routing=alternating_learn_subsequent_routing,
                                            alternating_learn_exponents=alternating_learn_exponents,
                                            alternating_min_exponent=alternating_min_exponent,
                                            alternating_max_exponent=alternating_max_exponent,
                                            alternating_max_egress=alternating_max_egress,
                                            alternating_use_straight_through=alternating_use_straight_through,
                                            alternating_balance_weight=loss_weight_alternating_balance,
                                            alternating_egress_weight=loss_weight_alternating_egress,
                                            use_constant_input=use_constant_input,
                                            use_permutation_layer=use_permutation_layer)
        
        if self.assembler.transformation_layer:
            actual_trans = self.assembler.transformation_layer.transformations
            logging.info(f"✅ Assembler created with {len(actual_trans)} transformations: {[type(t).__name__ for t in actual_trans]}")
    def forward(self, x):        
        output = self.assembler(x)
        return output
    def train_model(self, x, y, epochs):        
        try:
            output = self.assembler.train_model(x,y, epochs, self.is_frozen)
        except RuntimeError as e:
                # We'll raise the error now as there's only one binaryTreeLogicNet
            raise e
        return output

    def inference(self, x, threshold=0.5):        
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            # probs = torch.sigmoid(outputs)       # Convert to probabilities
            # predictions = (probs > 0.5).float() 
            predictions = (outputs >= threshold).float()
            return predictions
    def inference_raw(self, x):        
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs
    def evaluate(self, x, y, threshold=0.5):        
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            # probs = torch.sigmoid(outputs)       # Convert to probabilities
            # predictions = (probs > 0.5).float() 
            predictions = (outputs >= threshold).float()  # Binarize the output to match the target labels (0 or 1)
            accuracy = (predictions == y).float().mean()
            return accuracy.item()
        
    def save_model(self, filepath):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.assembler.save_model(filepath)

    def load_model(self, filepath):
        self.assembler.load_model(filepath)

    def make_param_groups(self):        
        param_groups = []
        
        # Group 1: Permutation layer (input_to_leaf)
        # Higher LR - needs to explore large combinatorial space (n! permutations)
        if hasattr(self.assembler, 'input_to_leaf') and hasattr(self.assembler.input_to_leaf, 'logits'):
            param_groups.append({
                'params': [self.assembler.input_to_leaf.logits],
                'lr': self.lr_permutation,
                'name': 'permutation'
            })
        
        # Group 2: Transformation layer (if exists)
        # Highest LR - simpler problem (2^n states), can move faster
        if self.assembler.transformation_layer is not None:
            trans_params = list(self.assembler.transformation_layer.parameters())
            if trans_params:
                param_groups.append({
                    'params': trans_params,
                    'lr': self.lr_transformation,
                    'name': 'transformation'
                })
        
        # Group 3: Aggregator weights (only if it has parameters)
        # Lower LR - tree structure should be more stable
        if hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'parameters'):
            try:
                agg_params = list(self.assembler.aggregator.parameters())
                if agg_params:
                    param_groups.append({
                        'params': agg_params,
                        'lr': self.lr_aggregator,
                        'name': 'aggregator'
                    })
            except:
                pass  # Some aggregators don't have learnable parameters
        
        # Group 4: Any other parameters (fallback)
        # Conservative LR for general weights
        registered_params = set()
        for group in param_groups:
            for p in group['params']:
                registered_params.add(id(p))
        
        other_params = [p for p in self.assembler.parameters() 
                       if id(p) not in registered_params]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.lr_other,
                'name': 'other'
            })
        
        return param_groups
    
    def _setup_training_attempt(self, y, coarse_perm, hierarchical_group_size, 
                                hierarchical_bleed_ratio, hierarchical_bleed_decay,
                                sinkhorn_iters, loss_weight_perm_sparsity,
                                freeze_aggregation_epochs, epochs_per_attempt, annealing_epochs,
                                task_type="classification",
                                operator_initial_tau=5.0, operator_final_tau=0.5,
                                operator_freeze_min_confidence=0.7,
                                operator_freeze_epochs=0):
        """Set up all components needed for a training attempt.
        
        Args:
            task_type: "classification" for BCE loss or "regression" for MSE loss
            operator_freeze_epochs: Epochs to freeze operator selection (phase 1)
        
        Returns:
            TrainingSetup: Dataclass containing all setup state
        """
        # Re-initialize assembler
        cfg = self.assembler

        # Create a fresh aggregator instance so NaN parameters from a failed
        # attempt don't poison subsequent attempts.
        fresh_aggregator = copy.deepcopy(self._original_aggregator)
        if hasattr(fresh_aggregator, 'to') and cfg.device is not None:
            fresh_aggregator = fresh_aggregator.to(cfg.device)

        self.assembler = binaryTreeLogicNet(
            cfg.original_input_size,
            weight_mode=cfg.weight_mode,
            weight_normalization=cfg.weight_normalization,
            weight_value=cfg.weight_value,
            weight_range=cfg.weight_range,
            weight_choices=None,
            noise_increase=cfg.noise_increase,
            noise_decrease=cfg.noise_decrease,
            loss_amplifier=cfg.loss_amplifier,
            normalize_andness=cfg.normalize_andness,
            min_noise=cfg.min_noise,
            max_noise=cfg.max_noise,
            is_frozen=False,
            tree_layout=cfg.tree_layout,
            weight_penalty_strength=cfg.weight_penalty_strength,
            aggregator=fresh_aggregator,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            early_stop_threshold=cfg.early_stop_threshold,
            use_transformation_layer=cfg.use_transformation_layer,
            transformation_temperature=self.transformation_initial_temperature,
            transformation_use_gumbel=cfg.transformation_layer.use_gumbel if cfg.transformation_layer else False,
            transformations=cfg._custom_transformations if hasattr(cfg, '_custom_transformations') else None,
            device=cfg.device,
            sinkhorn_iters=sinkhorn_iters,
            # Full tree parameters
            full_tree_depth=cfg.full_tree_depth,
            full_tree_shape=cfg.full_tree_shape,
            full_tree_temperature=cfg.full_tree_temperature,
            full_tree_final_temperature=cfg.full_tree_final_temperature,
            full_tree_max_egress=cfg.full_tree_max_egress,
            full_tree_concentrate_ingress=cfg.full_tree_concentrate_ingress,
            full_tree_use_sinkhorn=cfg.full_tree_use_sinkhorn,
            use_constant_input=cfg.use_constant_input,
            use_permutation_layer=cfg.use_permutation_layer,
            # Alternating tree parameters
            alternating_learn_first_routing=cfg.alternating_learn_first_routing,
            alternating_learn_subsequent_routing=cfg.alternating_learn_subsequent_routing,
            alternating_learn_exponents=cfg.alternating_learn_exponents,
            alternating_min_exponent=cfg.alternating_min_exponent,
            alternating_max_exponent=cfg.alternating_max_exponent,
            alternating_max_egress=cfg.alternating_max_egress,
            alternating_use_straight_through=cfg.alternating_use_straight_through,
            alternating_balance_weight=cfg.alternating_balance_weight,
            alternating_egress_weight=cfg.alternating_egress_weight,
        )
        
        # Initialize hierarchical permutation if provided
        if coarse_perm is not None:
            if hasattr(self.assembler, 'input_to_leaf') and hasattr(self.assembler.input_to_leaf, 'initialize_from_coarse_permutation'):
                self.assembler.input_to_leaf.initialize_from_coarse_permutation(
                    coarse_perm, 
                    group_size=hierarchical_group_size,
                    block_std=0.5,
                    bleed_ratio=hierarchical_bleed_ratio,
                    bleed_decay=hierarchical_bleed_decay
                )
                bleed_desc = "hard blocks" if hierarchical_bleed_ratio == 0 else f"bleed={hierarchical_bleed_ratio:.2f}"
                logging.info(f"   🎯 Initialized permutation matrix with coarse structure ({bleed_desc})")
        
        # Create parameter groups and optimizer
        param_groups = self.make_param_groups()
        optimizer = torch.optim.Adam(param_groups)
        
        # Log learning rates
        logging.info(f"   📚 Learning rates:")
        for group in param_groups:
            logging.info(f"      {group['name']}: {group['lr']}")
        
        # Log loss weighting
        total_weight = self.loss_weight_main + self.loss_weight_perm_entropy + self.loss_weight_trans_entropy + self.loss_weight_perm_sparsity
        if total_weight > 0:
            norm_main = self.loss_weight_main / total_weight
            norm_perm = self.loss_weight_perm_entropy / total_weight
            norm_trans = self.loss_weight_trans_entropy / total_weight
            norm_sparsity = self.loss_weight_perm_sparsity / total_weight
            if self.loss_weight_perm_entropy > 0 or self.loss_weight_trans_entropy > 0 or self.loss_weight_perm_sparsity > 0:
                logging.info(f"   ⚖️  Loss weights (normalized): main={norm_main:.3f}, perm_entropy={norm_perm:.3f}, trans_entropy={norm_trans:.3f}, perm_sparsity={norm_sparsity:.3f}")
        
        # Log full tree constraints if enabled
        if self.assembler.tree_layout == "full":
            if self.loss_weight_full_tree_ingress > 0 or self.loss_weight_full_tree_egress > 0 or self.full_tree_concentrate_ingress:
                ingress_mode = "column-softmax" if self.full_tree_concentrate_ingress else "soft penalty"
                logging.info(f"   🌳 Full tree constraints: ingress={self.loss_weight_full_tree_ingress:.2f} ({ingress_mode}), egress={self.loss_weight_full_tree_egress:.2f}")
        
        # Setup criterion based on task type
        pos_weight = None
        if task_type == "regression":
            criterion = nn.MSELoss()
            if self.regression_loss_type == "correlation":
                logging.info(f"   📈 Regression mode: using Pearson correlation loss (1 - r²)")
            elif self.regression_loss_type == "normalized_mse":
                logging.info(f"   📈 Regression mode: using normalized MSE (z-score both, then MSE)")
            else:
                logging.info(f"   📈 Regression mode: using MSE loss")
        elif self.use_class_weighting:
            pos_count = y.sum().item()
            neg_count = len(y) - pos_count
            if pos_count > 0:
                pos_weight = neg_count / pos_count
                logging.info(f"   ⚖️  Class weighting enabled: {pos_count} positives, {neg_count} negatives")
                logging.info(f"   ⚖️  Positive class weight: {pos_weight:.2f}x (penalizes defaults {pos_weight:.2f}x more)")
                criterion = nn.BCELoss(reduction='none')
            else:
                logging.warning(f"   ⚠️  Class weighting enabled but no positive samples found, using standard BCE")
                criterion = nn.BCELoss()
        else:
            logging.info(f"   ⚖️  Class weighting disabled: using standard BCE loss (original behavior)")
            criterion = nn.BCELoss()
        
        # Handle sparsity weight override
        original_sparsity_weight = None
        if loss_weight_perm_sparsity is not None:
            original_sparsity_weight = self.loss_weight_perm_sparsity
            self.loss_weight_perm_sparsity = loss_weight_perm_sparsity
            logging.info(f"   🎯 Using custom sparsity loss weight: {loss_weight_perm_sparsity}")
        
        # Log operator sparsity weight if applicable
        if hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'op_logits_per_node'):
            if self.loss_weight_operator_sparsity > 0:
                logging.info(f"   🔧 Operator sparsity weight: {self.loss_weight_operator_sparsity} (encourages commitment)")
            if self.loss_weight_operator_l2 > 0:
                logging.info(f"   🔧 Operator L2 weight: {self.loss_weight_operator_l2} (keeps logits bounded)")
        
        # Freeze aggregation if requested
        aggregation_frozen = False
        if freeze_aggregation_epochs > 0:
            for group in param_groups:
                if group['name'] in ['aggregator', 'other']:
                    for p in group['params']:
                        p.requires_grad = False
                    aggregation_frozen = True
            if aggregation_frozen:
                logging.info(f"   🧊 Aggregation frozen for first {freeze_aggregation_epochs} epochs (permutation-only learning)")
        
        # Temperature annealing setup
        use_temperature_annealing = (hasattr(self.assembler, 'input_to_leaf') and 
                                     hasattr(self.assembler.input_to_leaf, 'temperature') and
                                     hasattr(self.assembler.input_to_leaf, 'logits'))
        
        perm_temp_decay_rate = None
        trans_temp_decay_rate = None
        anneal_over_epochs_val = epochs_per_attempt
        
        if use_temperature_annealing:
            self.assembler.input_to_leaf.temperature = self.permutation_initial_temperature
            perm_initial_temp = self.permutation_initial_temperature
            perm_final_temp = self.permutation_final_temperature
            anneal_over_epochs_val = annealing_epochs if annealing_epochs else epochs_per_attempt
            perm_temp_decay_rate = (perm_final_temp / perm_initial_temp) ** (1.0 / anneal_over_epochs_val)
            logging.info(f"   🌡️  Permutation annealing: {perm_initial_temp:.1f} → {perm_final_temp:.1f} over {anneal_over_epochs_val} epochs (decay: {perm_temp_decay_rate:.6f})")
            if annealing_epochs and annealing_epochs < epochs_per_attempt:
                logging.info(f"   ⏱️  Frozen training: {epochs_per_attempt - anneal_over_epochs_val} epochs after hardening")
            
            if self.assembler.transformation_layer is not None:
                trans_initial_temp = self.transformation_initial_temperature
                trans_final_temp = self.transformation_final_temperature
                trans_temp_decay_rate = (trans_final_temp / trans_initial_temp) ** (1.0 / anneal_over_epochs_val)
                self.assembler.transformation_layer.temperature = trans_initial_temp
                logging.info(f"   🔗 Transformation annealing: {trans_initial_temp:.1f} → {trans_final_temp:.1f} over {anneal_over_epochs_val} epochs (decay: {trans_temp_decay_rate:.6f})")
        
        # Operator tau annealing setup (for OperatorSetAggregator)
        operator_tau_decay_rate = None
        if hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'tau'):
            self.assembler.aggregator.tau = operator_initial_tau
            # If using two-phase training, tau annealing happens only in phase 2
            # So we need to anneal over (anneal_over_epochs - operator_freeze_epochs) epochs
            effective_anneal_epochs = max(1, anneal_over_epochs_val - operator_freeze_epochs)
            operator_tau_decay_rate = (operator_final_tau / operator_initial_tau) ** (1.0 / effective_anneal_epochs)
            if operator_freeze_epochs > 0:
                logging.info(f"   🔧 Operator tau annealing: {operator_initial_tau:.1f} → {operator_final_tau:.1f} over epochs {operator_freeze_epochs}-{anneal_over_epochs_val} (decay: {operator_tau_decay_rate:.6f})")
            else:
                logging.info(f"   🔧 Operator tau annealing: {operator_initial_tau:.1f} → {operator_final_tau:.1f} over {anneal_over_epochs_val} epochs (decay: {operator_tau_decay_rate:.6f})")
            if operator_freeze_min_confidence > 0:
                logging.info(f"   🎯 Operator freeze threshold: {operator_freeze_min_confidence:.0%} confidence required before freezing")
        
        # Return dataclass with all setup state
        return TrainingSetup(
            optimizer=optimizer,
            criterion=criterion,
            pos_weight=pos_weight,
            param_groups=param_groups,
            aggregation_frozen=aggregation_frozen,
            actual_max_epochs=epochs_per_attempt,
            use_temperature_annealing=use_temperature_annealing,
            perm_temp_decay_rate=perm_temp_decay_rate,
            trans_temp_decay_rate=trans_temp_decay_rate,
            operator_tau_decay_rate=operator_tau_decay_rate,
            operator_initial_tau=operator_initial_tau if operator_tau_decay_rate else None,
            anneal_over_epochs=anneal_over_epochs_val,
            original_sparsity_weight=original_sparsity_weight,
            task_type=task_type,
            # Initialize tracking variables
            loss_history=[],
            accuracy_history=[],
            freeze_confidence_warning_shown=False,
            best_loss_for_convergence=float('inf'),
            epochs_without_improvement=0,
            epoch_when_frozen=None,
            has_converged_before_freeze=False,
            best_confidence=0.0,
            epochs_without_confidence_improvement=0,
            best_loss=float('inf'),
            epochs_since_improvement=0,
            temp_paused=False,
            transformation_converged=False,
            # Frozen training stability
            best_frozen_loss=None,
            best_frozen_state=None,
            frozen_lr_reduced=False,
            # Best overall state
            best_overall_loss=float('inf'),
            best_overall_state=None,
            best_overall_epoch=None
        )
    
    def _compute_composite_loss(self, outputs, y, criterion, pos_weight, epoch, 
                                sparsity_schedule, use_temperature_annealing, current_sparsity_weight,
                                task_type="classification"):
        """Compute composite loss with all regularization terms.
        
        Args:
            task_type: "classification" for BCE loss or "regression" for MSE loss
        """
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            logging.error(f"   ❌ NaN detected in model outputs at epoch {epoch + 1}")
            logging.error(f"      Output stats: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, mean={outputs.mean().item():.6f}")
            outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        # Compute per-sample loss based on task type
        if task_type == "regression":
            if self.regression_loss_type == "correlation":
                # Pearson correlation loss: 1 - r²
                # This is scale-invariant, helping when mul creates large output variations
                pred = outputs.view(-1)
                target = y.view(-1)
                
                # Compute means
                pred_mean = pred.mean()
                target_mean = target.mean()
                
                # Compute covariance and standard deviations
                pred_centered = pred - pred_mean
                target_centered = target - target_mean
                
                covariance = (pred_centered * target_centered).mean()
                pred_std = pred_centered.pow(2).mean().sqrt()
                target_std = target_centered.pow(2).mean().sqrt()
                
                # Pearson correlation (with stability epsilon)
                eps = 1e-8
                correlation = covariance / (pred_std * target_std + eps)
                
                # Loss is 1 - r² (so perfect correlation = 0 loss)
                correlation_loss = 1.0 - correlation.pow(2)
                
                # For correlation, we compute batch-level loss directly (no per-sample trimming)
                main_loss = correlation_loss * self.assembler.loss_amplifier
            elif self.regression_loss_type == "normalized_mse":
                # Normalized MSE: z-score both predictions and targets, then compute MSE
                # This is scale-invariant but stricter than correlation - requires matching patterns
                pred = outputs.view(-1)
                target = y.view(-1)
                
                eps = 1e-8
                
                # Z-score normalization
                pred_mean = pred.mean()
                pred_std = pred.std() + eps
                target_mean = target.mean()
                target_std = target.std() + eps
                
                pred_normalized = (pred - pred_mean) / pred_std
                target_normalized = (target - target_mean) / target_std
                
                # MSE on normalized values
                normalized_mse = F.mse_loss(pred_normalized, target_normalized)
                
                # For normalized MSE, compute batch-level loss directly
                main_loss = normalized_mse * self.assembler.loss_amplifier
            else:
                per_sample = F.mse_loss(outputs, y, reduction='none').view(-1)
        else:
            per_sample_bce = F.binary_cross_entropy(outputs, y, reduction='none').view(-1)
            # Optional class weighting (only for classification)
            if self.use_class_weighting and pos_weight is not None:
                weights = torch.where(y.view(-1) == 1, torch.as_tensor(pos_weight, device=y.device, dtype=per_sample_bce.dtype), torch.ones_like(per_sample_bce))
                per_sample = per_sample_bce * weights
            else:
                per_sample = per_sample_bce
        
        # For correlation/normalized_mse loss, main_loss is already computed; otherwise compute from per_sample
        if not (task_type == "regression" and self.regression_loss_type in ("correlation", "normalized_mse")):
            # Optional loss trimming by percentile
            # Apply warmup schedule: no trimming before start epoch
            trim_p = self.loss_trim_percentile if epoch >= self.loss_trim_start_epoch else 0.0
            if trim_p > 0.0 and self.loss_trim_mode != "none" and per_sample.numel() > 1:
                # Compute threshold
                q = 1.0 - trim_p if self.loss_trim_mode == "drop_high" else trim_p
                thresh = torch.quantile(per_sample.detach(), q)
                if self.loss_trim_mode == "drop_high":
                    mask = per_sample <= thresh
                else:
                    mask = per_sample >= thresh
                if mask.any():
                    per_sample = per_sample[mask]
            # Final main loss
            main_loss = per_sample.mean() * self.assembler.loss_amplifier
        
        if torch.isnan(main_loss):
            logging.error(f"   ❌ NaN in main_loss at epoch {epoch + 1}")
            logging.error(f"      BCE output: {main_loss.item()}")
        
        loss = self.loss_weight_main * main_loss
        
        if torch.isnan(loss):
            logging.error(f"   ❌ NaN in composite loss (after main_loss) at epoch {epoch + 1}")
        
        # Add permutation entropy regularization
        if self.loss_weight_perm_entropy > 0 and hasattr(self.assembler.input_to_leaf, 'logits'):
            perm_probs = torch.softmax(self.assembler.input_to_leaf.logits / self.assembler.input_to_leaf.temperature, dim=1)
            perm_probs_clamped = torch.clamp(perm_probs, min=1e-8, max=1.0 - 1e-8)
            perm_entropy = -(perm_probs_clamped * torch.log(perm_probs_clamped)).sum(dim=1).mean()
            loss = loss - self.loss_weight_perm_entropy * perm_entropy
        
        # Add transformation entropy regularization
        if self.loss_weight_trans_entropy > 0 and self.assembler.transformation_layer is not None:
            trans_probs = torch.softmax(self.assembler.transformation_layer.logits / self.assembler.transformation_layer.temperature, dim=1)
            trans_probs_clamped = torch.clamp(trans_probs, min=1e-8, max=1.0 - 1e-8)
            trans_entropy = -(trans_probs_clamped * torch.log(trans_probs_clamped)).sum(dim=1).mean()
            loss = loss - self.loss_weight_trans_entropy * trans_entropy
        
        # Dynamic sparsity weight scheduling
        if sparsity_schedule is not None and use_temperature_annealing:
            initial_weight, final_weight, transition_epochs = sparsity_schedule
            if epoch < transition_epochs:
                alpha = epoch / transition_epochs
                current_sparsity_weight = initial_weight * (1 - alpha) + final_weight * alpha
            else:
                current_sparsity_weight = final_weight
        
        # Add permutation sparsity loss
        if current_sparsity_weight > 0 and use_temperature_annealing and hasattr(self.assembler.input_to_leaf, 'logits'):
            if hasattr(self.assembler.input_to_leaf, 'sinkhorn'):
                perm_probs = self.assembler.input_to_leaf.sinkhorn(
                    self.assembler.input_to_leaf.logits,
                    temperature=self.assembler.input_to_leaf.temperature,
                    n_iters=self.assembler.input_to_leaf.sinkhorn_iters
                )
            else:
                perm_probs = torch.softmax(self.assembler.input_to_leaf.logits, dim=1)
            
            perm_probs_clamped = torch.clamp(perm_probs, min=1e-8, max=1.0 - 1e-8)
            perm_sparsity_entropy = -(perm_probs_clamped * torch.log(perm_probs_clamped)).sum(dim=1).mean()
            
            if torch.isnan(perm_sparsity_entropy):
                logging.error(f"   ❌ NaN in perm_sparsity_entropy at epoch {epoch + 1}")
                logging.error(f"      perm_probs stats: min={perm_probs.min().item():.6f}, max={perm_probs.max().item():.6f}")
                perm_sparsity_entropy = torch.tensor(0.0, device=perm_probs.device)
            
            sparsity_loss = current_sparsity_weight * perm_sparsity_entropy
            loss = loss + sparsity_loss
            
            if torch.isnan(loss):
                logging.error(f"   ❌ NaN in loss after adding sparsity at epoch {epoch + 1}")
            
            if not hasattr(self.assembler.input_to_leaf, 'sinkhorn'):
                col_sums = perm_probs.sum(dim=0)
                col_uniqueness_penalty = ((col_sums - 1.0) ** 2).mean()
                loss = loss + current_sparsity_weight * col_uniqueness_penalty
        
        # Operator selection sparsity regularization (for OperatorSetAggregator)
        # Penalizes uncertain operator selections to encourage commitment
        if self.loss_weight_operator_sparsity > 0 and hasattr(self.assembler, 'aggregator'):
            aggregator = self.assembler.aggregator
            if hasattr(aggregator, 'op_logits_per_node') and aggregator.op_logits_per_node is not None:
                op_entropy_total = 0.0
                for logits in aggregator.op_logits_per_node:
                    # Get current tau for temperature-adjusted probabilities
                    tau = getattr(aggregator, 'tau', 1.0)
                    probs = torch.softmax(logits / tau, dim=0)
                    probs_clamped = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
                    # Entropy: high when uncertain, low when committed
                    entropy = -(probs_clamped * torch.log(probs_clamped)).sum()
                    op_entropy_total += entropy
                # Normalize by number of nodes
                num_nodes = len(aggregator.op_logits_per_node)
                if num_nodes > 0:
                    op_sparsity_loss = self.loss_weight_operator_sparsity * (op_entropy_total / num_nodes)
                    loss = loss + op_sparsity_loss
        
        # Operator logit L2 regularization - keeps logits bounded so tau can control commitment
        # This prevents premature operator commitment by keeping softmax close to uniform when tau is high
        if self.loss_weight_operator_l2 > 0 and hasattr(self.assembler, 'aggregator'):
            aggregator = self.assembler.aggregator
            if hasattr(aggregator, 'op_logits_per_node') and aggregator.op_logits_per_node is not None:
                l2_total = 0.0
                for logits in aggregator.op_logits_per_node:
                    l2_total += (logits ** 2).mean()
                num_nodes = len(aggregator.op_logits_per_node)
                if num_nodes > 0:
                    op_l2_loss = self.loss_weight_operator_l2 * (l2_total / num_nodes)
                    loss = loss + op_l2_loss
        
        # Weight regularization
        if self.assembler.weight_mode == "trainable":
            depth_weight_penalty = 0.0
            for w in self.assembler.weights:
                depth_weight_penalty += self.assembler.weight_penalty_strength * ((torch.sigmoid(w) - 0.5) ** 2).mean()
            loss = loss + depth_weight_penalty
        
        # Full tree regularization (when tree_layout="full")
        if self.assembler.tree_layout == "full":
            # Get current R² if available (used to modulate structural penalties)
            # High R² + bad structure = coefficient hacking, needs stronger penalty
            r2_factor = 1.0
            if hasattr(self, '_current_r2') and self._current_r2 is not None:
                # Scale factor increases with R²: at R²=0 factor=1, at R²=1 factor=3
                r2_factor = 1.0 + 2.0 * max(0.0, self._current_r2)
            
            # Egress constraint loss: encourage each source to concentrate outgoing edges to top-K
            egress_weight = getattr(self, '_dynamic_full_tree_egress_weight', self.loss_weight_full_tree_egress)
            if egress_weight > 0:
                full_tree_egress = self.assembler.get_full_tree_egress_loss()
                loss = loss + egress_weight * full_tree_egress
            
            # Ingress constraint loss: discourage more than 2 inputs per aggregator node
            if self.loss_weight_full_tree_ingress > 0:
                full_tree_ingress = self.assembler.get_full_tree_ingress_loss()
                ingress_penalty = self.loss_weight_full_tree_ingress * full_tree_ingress
                loss = loss + ingress_penalty
                # Debug: log ingress penalty every 500 epochs
                if hasattr(self, '_epoch_for_logging') and self._epoch_for_logging % 500 == 0:
                    logging.debug(f"      Ingress penalty: {ingress_penalty.item():.4f} (raw: {full_tree_ingress.item():.4f})")
            
            # Ingress balance loss: prevent all sources from routing to same destination
            # R²-modulated: higher R² with imbalanced structure = harsher penalty
            if self.loss_weight_full_tree_ingress_balance > 0:
                balance_loss = self.assembler.get_full_tree_ingress_balance_loss()
                loss = loss + self.loss_weight_full_tree_ingress_balance * r2_factor * balance_loss
            
            # Scale regularization: penalize extreme coefficients to prevent coefficient hacking
            # R²-modulated: high R² with extreme scales = likely hacking
            if self.loss_weight_full_tree_scale_reg > 0:
                scale_reg = self.assembler.get_full_tree_scale_regularization_loss()
                loss = loss + self.loss_weight_full_tree_scale_reg * r2_factor * scale_reg
        
        # Alternating tree structural losses
        if self.assembler.tree_layout == "alternating" and self.assembler.alternating_tree is not None:
            # Balance loss: prevent all sources from routing to same destination
            if self.loss_weight_alternating_balance > 0:
                balance_loss = self.assembler.get_alternating_tree_balance_loss()
                loss = loss + self.loss_weight_alternating_balance * balance_loss
            
            # Egress loss: encourage peaked routing distributions
            if self.loss_weight_alternating_egress > 0:
                egress_loss = self.assembler.get_alternating_tree_egress_loss()
                loss = loss + self.loss_weight_alternating_egress * egress_loss

            if self.loss_weight_alternating_exponent_reg > 0:
                exponent_reg = self.assembler.get_alternating_tree_exponent_regularization_loss()
                loss = loss + self.loss_weight_alternating_exponent_reg * exponent_reg
        
        # Training policy regularization (e.g., andness penalty)
        if self.training_policy is not None and hasattr(self.training_policy, 'penalty'):
            policy_penalty = self.training_policy.penalty(self.assembler)
            loss = loss + policy_penalty
        
        return loss
    
    def _check_and_freeze_permutation(self, epoch, x_test, y_test, criterion, 
                                     freeze_confidence_threshold, freeze_min_confidence,
                                     early_stop_threshold, perm_final_temp,
                                     has_converged_before_freeze, best_confidence,
                                     epochs_without_confidence_improvement, confidence_plateau_patience,
                                     freeze_confidence_warning_shown, current_loss, binary_threshold=0.5,
                                     task_type="classification", operator_freeze_min_confidence=0.0,
                                     skip_frozen_threshold=0.95):
        """Check if permutation should be frozen and perform freezing if ready.
        
        Args:
            operator_freeze_min_confidence: Minimum operator selection confidence required before freezing.
                                           0.0 = no requirement (legacy behavior), 0.7 = require 70% average.
            skip_frozen_threshold: Minimum metric (R² or accuracy) required to skip frozen training.
                                  Only skip if freeze improves metric AND we're above this threshold.
                                  Default 0.95 means skip only if already excellent.
        
        Returns:
            tuple: (just_froze, best_confidence, epochs_without_confidence_improvement, freeze_confidence_warning_shown, skip_frozen_training)
        """
        just_froze = False
        skip_frozen_training = False
        
        if not self.assembler.is_frozen and hasattr(self.assembler.input_to_leaf, 'logits'):
            try:
                from bacon.frozonInputToLeaf import frozenInputToLeaf
                from scipy.optimize import linear_sum_assignment
                
                # Get soft permutation matrix
                with torch.no_grad():
                    if hasattr(self.assembler.input_to_leaf, 'sinkhorn'):
                        soft_perm = self.assembler.input_to_leaf.sinkhorn(
                            self.assembler.input_to_leaf.logits,
                            temperature=self.assembler.input_to_leaf.temperature,
                            n_iters=self.assembler.input_to_leaf.sinkhorn_iters
                        )
                    else:
                        soft_perm = torch.softmax(self.assembler.input_to_leaf.logits, dim=1)
                    
                    significant_entries = (soft_perm > 0.01).sum(dim=1).float().mean().item()
                
                # Check confidence
                max_probs = soft_perm.max(dim=1)[0]
                mean_confidence = max_probs.mean().item()
                
                # Track confidence improvement
                if mean_confidence > best_confidence + 0.001:
                    best_confidence = mean_confidence
                    epochs_without_confidence_improvement = 0
                else:
                    epochs_without_confidence_improvement += 1
                
                # Check operator confidence if aggregator supports it
                operator_confidence_ok = True  # Default to True for aggregators without operator selection
                current_op_confidence = 0.0
                if operator_freeze_min_confidence > 0 and hasattr(self.assembler, 'aggregator'):
                    aggregator = self.assembler.aggregator
                    if hasattr(aggregator, 'tau') and hasattr(aggregator, 'op_logits_per_node') and aggregator.op_logits_per_node is not None:
                        with torch.no_grad():
                            confidences = []
                            for logits in aggregator.op_logits_per_node:
                                probs = torch.softmax(logits / aggregator.tau, dim=0)
                                confidences.append(probs.max().item())
                            current_op_confidence = sum(confidences) / len(confidences) if confidences else 1.0
                        operator_confidence_ok = current_op_confidence >= operator_freeze_min_confidence
                
                # Freeze conditions
                confidence_reached = mean_confidence >= freeze_confidence_threshold
                loss_threshold = early_stop_threshold * self.assembler.loss_amplifier * 1.5
                good_confidence_and_loss = (mean_confidence >= freeze_min_confidence and current_loss < loss_threshold)
                confidence_plateaued = (has_converged_before_freeze and 
                                       epochs_without_confidence_improvement >= confidence_plateau_patience)
                
                perm_should_freeze = confidence_reached or good_confidence_and_loss or confidence_plateaued
                should_freeze = perm_should_freeze and operator_confidence_ok
                
                if not should_freeze:
                    temp_ready_to_freeze = self.assembler.input_to_leaf.temperature < perm_final_temp * 1.1
                    if temp_ready_to_freeze and not freeze_confidence_warning_shown:
                        logging.info(f"   ⚠️  Permutation temp low ({self.assembler.input_to_leaf.temperature:.3f}) but not ready to freeze")
                        logging.info(f"      Permutation confidence: {mean_confidence:.3f} (target: {freeze_confidence_threshold:.2f})")
                        if not operator_confidence_ok:
                            logging.info(f"      Operator confidence: {current_op_confidence:.3f} (target: {operator_freeze_min_confidence:.2f}) ← blocking freeze")
                        if has_converged_before_freeze:
                            logging.info(f"      Loss converged, waiting for confidence plateau ({epochs_without_confidence_improvement}/{confidence_plateau_patience} epochs)")
                        else:
                            logging.info(f"      Waiting for loss convergence first...")
                        freeze_confidence_warning_shown = True
                    just_froze = True
                    return just_froze, best_confidence, epochs_without_confidence_improvement, freeze_confidence_warning_shown, skip_frozen_training
                
                # Ready to freeze!
                if confidence_plateaued and not confidence_reached:
                    logging.info(f"   ⚠️  Confidence plateaued at {mean_confidence:.3f} (below target {freeze_confidence_threshold:.2f})")
                    logging.info(f"      Freezing anyway after {epochs_without_confidence_improvement} epochs without improvement")
                
                logging.info(f"   ❄️  Permutation hardened at epoch {epoch + 1} (temp: {self.assembler.input_to_leaf.temperature:.3f})")
                logging.info(f"      Mean confidence: {mean_confidence:.3f} ✓")
                
                if significant_entries > 1.5:
                    logging.info(f"      Detected hierarchical structure (avg {significant_entries:.1f} entries/row)")
                    logging.info(f"      Using Hungarian algorithm for optimal assignment...")
                else:
                    logging.info(f"      Detected standard permutation (avg {significant_entries:.1f} entries/row), using Hungarian algorithm")
                
                # DIAGNOSTIC: Before freeze
                self.assembler.eval()
                with torch.no_grad():
                    before_output = self.assembler(x_test)
                    before_loss_raw = criterion(before_output, y_test)
                    before_loss = before_loss_raw.mean() if before_loss_raw.dim() > 0 else before_loss_raw
                    if task_type == "regression":
                        ss_res = ((y_test - before_output) ** 2).sum()
                        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                        before_metric = (1 - ss_res / (ss_tot + 1e-8)).item()
                        metric_name = "R²"
                    else:
                        before_pred = (before_output > binary_threshold).float()
                        before_metric = (before_pred == y_test).float().mean().item()
                        metric_name = "Accuracy"
                    
                    max_probs = soft_perm.max(dim=1)[0]
                    logging.info(f"      📊 BEFORE FREEZE (test set):")
                    logging.info(f"         Loss: {before_loss.item():.4f}, Test {metric_name}: {before_metric:.4f}")
                    logging.info(f"         Soft perm max probs: min={max_probs.min():.3f}, mean={max_probs.mean():.3f}, max={max_probs.max():.3f}")
                    
                    for i in range(min(5, soft_perm.size(0))):
                        top_vals, top_idx = soft_perm[i].topk(3)
                        logging.info(f"         Row {i}: top3 = {top_vals.cpu().numpy()} at positions {top_idx.cpu().numpy()}")
                
                self.assembler.train()
                
                # Verify doubly-stochastic property
                if hasattr(self.assembler.input_to_leaf, 'sinkhorn'):
                    row_sums = soft_perm.sum(dim=1)
                    col_sums = soft_perm.sum(dim=0)
                    row_sum_error = (row_sums - 1.0).abs().max().item()
                    col_sum_error = (col_sums - 1.0).abs().max().item()
                    if row_sum_error > 0.01 or col_sum_error > 0.01:
                        logging.warning(f"      ⚠️  Sinkhorn not doubly-stochastic: row_error={row_sum_error:.4f}, col_error={col_sum_error:.4f}")
                        logging.warning(f"         Current Sinkhorn iterations: {self.assembler.input_to_leaf.sinkhorn_iters}")
                
                # Check for duplicates
                argmax_perm = soft_perm.argmax(dim=1)
                unique_cols = len(torch.unique(argmax_perm))
                total_rows = argmax_perm.size(0)
                if unique_cols < total_rows:
                    logging.info(f"      ⚠️  Argmax creates duplicate columns: {unique_cols} unique out of {total_rows} rows")
                
                # Use Hungarian algorithm as baseline
                soft_perm_np = soft_perm.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-soft_perm_np)
                hungarian_perm = torch.tensor(col_ind[row_ind.argsort()], dtype=torch.long, device=self.assembler.device)
                
                # Sample multiple candidate permutations for uncertain rows
                logging.info(f"      🎲 Sampling candidate permutations to find best freeze assignment...")
                max_probs = soft_perm.max(dim=1)[0]
                uncertain_threshold = 0.90  # Rows with confidence < 90% are considered uncertain
                uncertain_rows = (max_probs < uncertain_threshold).nonzero(as_tuple=True)[0]
                
                if len(uncertain_rows) > 0:
                    logging.info(f"      Found {len(uncertain_rows)} uncertain rows (confidence < {uncertain_threshold:.0%})")
                    
                    # Generate candidate permutations by exploring top-k assignments for uncertain rows
                    candidates = [hungarian_perm]  # Start with Hungarian as baseline
                    max_candidates = 20  # Limit total candidates to avoid explosion
                    
                    # For each uncertain row, get top-2 or top-3 choices
                    uncertain_choices = {}
                    for row_idx in uncertain_rows:
                        topk = min(3, soft_perm.size(1))  # Top-3 choices
                        top_vals, top_indices = soft_perm[row_idx].topk(topk)
                        uncertain_choices[row_idx.item()] = top_indices.tolist()
                    
                    # Generate permutations by varying uncertain row assignments
                    # Use a greedy strategy: try swapping each uncertain row one at a time
                    for row_idx, choices in uncertain_choices.items():
                        for choice in choices[1:]:  # Skip first choice (already in Hungarian)
                            # Create candidate by modifying Hungarian
                            candidate = hungarian_perm.clone()
                            old_assignment = candidate[row_idx]
                            candidate[row_idx] = choice
                            
                            # Check if this creates a duplicate (column conflict)
                            if len(torch.unique(candidate)) == len(candidate):
                                candidates.append(candidate)
                                if len(candidates) >= max_candidates:
                                    break
                        if len(candidates) >= max_candidates:
                            break
                    
                    logging.info(f"      Generated {len(candidates)} candidate permutations (including Hungarian baseline)")
                    
                    # Evaluate all candidates on TRAINING set to avoid test set overfitting
                    best_train_acc = -1
                    best_perm = hungarian_perm
                    best_perm_idx = 0
                    
                    self.assembler.eval()
                    with torch.no_grad():
                        for i, candidate_perm in enumerate(candidates):
                            # Temporarily set this permutation
                            temp_frozen_input = frozenInputToLeaf(candidate_perm, self.assembler.original_input_size).to(self.assembler.device)
                            original_input_to_leaf = self.assembler.input_to_leaf
                            self.assembler.input_to_leaf = temp_frozen_input
                            self.assembler.is_frozen = True
                            
                            # Evaluate on training set
                            train_output = self.assembler(x_test)  # Note: x_test here is actually training data passed to freeze check
                            train_pred = (train_output > 0.5).float()
                            train_acc = (train_pred == y_test).float().mean().item()
                            
                            if train_acc > best_train_acc:
                                best_train_acc = train_acc
                                best_perm = candidate_perm
                                best_perm_idx = i
                            
                            # Restore
                            self.assembler.input_to_leaf = original_input_to_leaf
                            self.assembler.is_frozen = False
                    
                    self.assembler.train()
                    
                    if best_perm_idx == 0:
                        logging.info(f"      ✓ Hungarian baseline is best (train acc: {best_train_acc:.4f})")
                    else:
                        logging.info(f"      ✨ Found better permutation #{best_perm_idx} (train acc: {best_train_acc:.4f} vs Hungarian: {candidates[0] and 'N/A'})")
                        differences = (best_perm != hungarian_perm).sum().item()
                        logging.info(f"         Differs from Hungarian in {differences}/{len(best_perm)} positions")
                    
                    hard_perm = best_perm
                else:
                    logging.info(f"      All rows have high confidence (>= {uncertain_threshold:.0%}), using Hungarian algorithm")
                    hard_perm = hungarian_perm
                
                perm_differences = (argmax_perm != hard_perm).sum().item()
                if perm_differences > 0:
                    logging.info(f"      ⚠️  Final permutation differs from argmax in {perm_differences}/{len(hard_perm)} positions")
                
                self.assembler.locked_perm = hard_perm.clone().detach()
                self.assembler.is_frozen = True
                
                if self.assembler.locked_perm is not None:
                    self.assembler.input_to_leaf = frozenInputToLeaf(
                        self.assembler.locked_perm,
                        self.assembler.original_input_size
                    ).to(self.assembler.device)
                
                # DIAGNOSTIC: After freeze
                self.assembler.eval()
                with torch.no_grad():
                    after_output = self.assembler(x_test)
                    after_loss_raw = criterion(after_output, y_test)
                    after_loss = after_loss_raw.mean() if after_loss_raw.dim() > 0 else after_loss_raw
                    if task_type == "regression":
                        ss_res = ((y_test - after_output) ** 2).sum()
                        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                        after_metric = (1 - ss_res / (ss_tot + 1e-8)).item()
                    else:
                        after_pred = (after_output > binary_threshold).float()
                        after_metric = (after_pred == y_test).float().mean().item()
                    
                    logging.info(f"      📊 AFTER FREEZE:")
                    logging.info(f"         Loss: {after_loss.item():.4f} (Δ={after_loss.item()-before_loss.item():+.4f})")
                    logging.info(f"         Test {metric_name}: {after_metric:.4f} (Δ={after_metric-before_metric:+.4f})")
                    
                    if task_type != "regression":
                        before_pred = (before_output > binary_threshold).float()
                        after_pred = (after_output > binary_threshold).float()
                        pred_changes = (before_pred != after_pred).sum().item()
                        if pred_changes > 0:
                            logging.info(f"         ⚠️  {pred_changes}/{len(y_test)} predictions changed after freeze")
                    
                    output_diff = (after_output - before_output).abs().max().item()
                    logging.info(f"         Max output difference: {output_diff:.6f}")
                
                self.assembler.train()
                logging.info(f"   🔒 Successfully frozen model (locked_perm created)")
                
                # Check if freeze improved metric - skip frozen training only if already excellent
                if after_metric > before_metric:
                    metric_gain = (after_metric - before_metric) * 100
                    if after_metric >= skip_frozen_threshold:
                        logging.info(f"   ✅ Freeze improved {metric_name} by {metric_gain:.2f}% to {after_metric:.4f} (≥{skip_frozen_threshold:.2f}) - stopping immediately")
                        logging.info(f"      Rationale: Already above threshold, further training may degrade performance")
                        just_froze = True
                        return just_froze, best_confidence, epochs_without_confidence_improvement, freeze_confidence_warning_shown, True  # skip_frozen_training=True
                    else:
                        logging.info(f"   📈 Freeze improved {metric_name} by {metric_gain:.2f}% to {after_metric:.4f} (below {skip_frozen_threshold:.2f})")
                        logging.info(f"      Continuing with frozen training to refine weights...")
                
                just_froze = True
                
            except Exception as e:
                logging.warning(f"   ⚠️ Failed to freeze on hardening: {e}")
                self.assembler.is_frozen = True
                just_froze = True
        
        return just_froze, best_confidence, epochs_without_confidence_improvement, freeze_confidence_warning_shown, skip_frozen_training

    def find_best_model(self, x, y, x_test, y_test, 
                        attempts = 100, 
                        acceptance_threshold = 0.95, 
                        save_path = "./assembler.pth", 
                        max_epochs = 12000,
                        annealing_epochs = None,
                        frozen_training_epochs = 200,
                        convergence_patience = 500,
                        convergence_delta = 0.001,
                        freeze_confidence_threshold = 0.95,
                        freeze_min_confidence = 0.85,
                        loss_weight_perm_sparsity = None,
                        sparsity_schedule = None,
                        freeze_aggregation_epochs = 0,
                        save_model = True,
                        use_hierarchical_permutation = False,
                        force_freeze = True,
                        hierarchical_group_size = 3,
                        hierarchical_epochs_per_attempt = None,
                        hierarchical_bleed_ratio = 0.1,
                        hierarchical_bleed_decay = 2.0,
                        sinkhorn_iters = 100,
                        binary_threshold=0.5,
                        task_type="classification",
                        operator_initial_tau=5.0,
                        operator_final_tau=0.5,
                        operator_freeze_min_confidence=0.7,
                        operator_freeze_epochs=0,
                        skip_frozen_threshold=0.99,
                        full_tree_egress_warmup_epochs=0,
                        full_tree_egress_ramp_epochs=0,
                        full_tree_egress_start_metric=0.99,
                        full_tree_egress_drop_tolerance=0.02,
                        full_tree_egress_adapt_rate=0.2):
        """ Find the best model by training multiple times and evaluating accuracy.

        Args:
            x (torch.Tensor): Input tensor for training.
            y (torch.Tensor): Target tensor for training.
            x_test (torch.Tensor): Input tensor for testing.
            y_test (torch.Tensor): Target tensor for testing.
            attempts (int, optional): Number of attempts to find the best model. Defaults to 100.
            acceptance_threshold (float, optional): Minimum accuracy to accept a model. Defaults to 0.95.
            save_path (str, optional): Path to save the best model. Defaults to "./assembler.pth".
            max_epochs (int, optional): Maximum epochs for training (safety limit). Defaults to 12000.
            annealing_epochs (int, optional): Epochs for temperature annealing. Defaults to None.
            frozen_training_epochs (int, optional): Epochs to train after freezing. Defaults to 200.
            convergence_patience (int, optional): Epochs without improvement before considering converged. Defaults to 500.
            convergence_delta (float, optional): Minimum loss improvement to reset patience. Defaults to 0.001.
            freeze_confidence_threshold (float, optional): Mean max probability threshold for high-confidence freezing. Defaults to 0.95.
            freeze_min_confidence (float, optional): Minimum confidence for early freeze when combined with low loss. Defaults to 0.85 (raised from 0.75 to be more conservative).
            sinkhorn_iters (int, optional): Number of Sinkhorn normalization iterations for soft permutation convergence. Higher values improve doubly-stochastic property but increase compute time. Defaults to 100.
            loss_weight_perm_sparsity (float, optional): Weight for permutation sparsity loss (encourages peaked distributions). If None, uses instance default. Defaults to None.
            sparsity_schedule (tuple, optional): Dynamic sparsity weight scheduling as (initial_weight, final_weight, transition_epochs). Example: (10.0, 0.1, 1000) starts with high sparsity emphasis (10.0) and linearly decreases to 0.1 over 1000 epochs. Defaults to None (uses constant loss_weight_perm_sparsity).
            freeze_aggregation_epochs (int, optional): Freeze aggregation parameters for first N epochs, allowing only permutation to learn. Useful for giving permutation undiluted classification signal. Defaults to 0 (no freezing).
            save_model (bool, optional): Whether to save the best model. Defaults to True.
            use_hierarchical_permutation (bool, optional): Use coarse-grained permutation exploration. Defaults to False.
            hierarchical_group_size (int, optional): Group size for hierarchical permutation (e.g., 3 for 10 inputs → 4x4 coarse matrix). Defaults to 3.
            hierarchical_epochs_per_attempt (int, optional): Epochs to run for each coarse permutation. If None, uses max_epochs. Defaults to None.
            hierarchical_bleed_ratio (float, optional): Ratio of std for adjacent blocks (0.0=hard blocks, 0.1=10% bleed, 1.0=full bleed). Defaults to 0.1.
            hierarchical_bleed_decay (float, optional): How quickly bleeding decays with distance (higher=faster decay). Defaults to 2.0.
            operator_freeze_min_confidence (float, optional): Minimum average operator selection confidence required before 
                freezing permutation. Blocks freeze until operators commit to their choices. 0.0 disables the requirement 
                (legacy behavior), 0.7 requires 70% average operator confidence. Defaults to 0.7.
            operator_freeze_epochs (int, optional): Freeze operator selection for first N epochs, allowing only 
                edge/routing to learn. This decouples structure discovery from operator selection. After N epochs,
                operators unfreeze and start learning. Defaults to 0 (no operator freeze).
            skip_frozen_threshold (float, optional): Minimum metric required to skip frozen training after freeze.
                Only skips frozen training if freeze improves metric AND we're above this threshold.
                For regression tasks, this should be very high (e.g., 0.99) to ensure weights fully converge.
                Defaults to 0.99.
            full_tree_egress_warmup_epochs (int, optional): In full-tree mode, keep egress concentration disabled
                for the first N epochs so structure can be learned first. Defaults to 0.
            full_tree_egress_ramp_epochs (int, optional): Number of epochs to ramp egress loss weight from 0
                to target after warmup. Defaults to 0 (immediate application after warmup).
            full_tree_egress_start_metric (float, optional): Minimum training metric required before egress
                concentration begins. This lets the full tree learn the task first before sparsifying
                edges. Defaults to 0.99.
            full_tree_egress_drop_tolerance (float, optional): Maximum allowed training-metric drop from warmup
                baseline before reducing egress pressure. Defaults to 0.02.
            full_tree_egress_adapt_rate (float, optional): Fractional backoff/adjustment rate for dynamic egress
                weight when metric drop exceeds tolerance. Defaults to 0.2.
        Returns:
            tuple: Best model state dictionary and its metric (accuracy or R²).
        """
        best_metric = -float('inf') if task_type == "regression" else 0.0
        best_model = None
        loaded_metric = -float('inf') if task_type == "regression" else 0.0

        if save_model and save_path and os.path.exists(save_path):
            try:
                logging.info(f"📂 Found saved model at {save_path}, loading...")
                self.load_model(save_path)
                
                # Debug: Check what was loaded
                if hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'op_logits_per_node'):
                    if self.assembler.aggregator.op_logits_per_node is not None:
                        for i, logits in enumerate(self.assembler.aggregator.op_logits_per_node):
                            probs = torch.softmax(logits, dim=0)
                            logging.info(f"   📋 After load - Node {i} op probs: {probs.tolist()}")
                
                if self.assembler.is_frozen:
                    if task_type == "regression":
                        output = self.inference_raw(x_test)
                        ss_res = ((y_test - output) ** 2).sum()
                        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                        metric = (1 - ss_res / (ss_tot + 1e-8)).item()  # R²
                        metric_name = "R²"
                    elif binary_threshold >= 0:
                        metric = self.evaluate(x_test, y_test, threshold=binary_threshold)
                        metric_name = "accuracy"
                    else:
                        output = self.inference_raw(x_test)
                        mae = (output - y_test).abs().mean().item()
                        metric = 1.0 - min(mae, 1.0)
                        metric_name = "accuracy"
                    logging.info(f"✅ Loaded model {metric_name}: {metric:.4f}")
                    loaded_metric = metric
                    if metric >= acceptance_threshold:
                        return self.assembler.state_dict(), metric
                    else:
                        logging.info(f"⚠️ Loaded {metric_name} {metric:.4f} < threshold {acceptance_threshold:.4f}, will retrain")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load model from {save_path}: {e}")

        # Determine how many attempts to run and what permutations to use
        if use_hierarchical_permutation and hasattr(self.assembler, 'input_to_leaf'):
            from bacon.inputToLeafSinkhorn import inputToLeafSinkhorn
            n = self.assembler.original_input_size
            coarse_perms = inputToLeafSinkhorn.generate_all_coarse_permutations(n, hierarchical_group_size)
            total_attempts = len(coarse_perms)
            epochs_per_attempt = hierarchical_epochs_per_attempt if hierarchical_epochs_per_attempt else max_epochs
            logging.info(f"🔀 Hierarchical permutation mode: {total_attempts} coarse permutations (group_size={hierarchical_group_size})")
            logging.info(f"   Each coarse permutation will train for {epochs_per_attempt} epochs")
        else:
            coarse_perms = [None] * attempts  # No hierarchical structure, use random init
            total_attempts = attempts
            epochs_per_attempt = max_epochs

        # Track best frozen state across all attempts
        best_is_frozen = False
        best_locked_perm = None
        best_full_tree_hardened = False
        best_alternating_hardened = False
        best_operator_hardened = False

        for attempt in range(total_attempts):
            if use_hierarchical_permutation and coarse_perms[attempt] is not None:
                logging.info(f"🔥 Attempting coarse permutation {attempt + 1}/{total_attempts}: {coarse_perms[attempt]}")
            else:
                logging.info(f"🔥 Attempting to find the best model... {attempt + 1}/{total_attempts}")

            torch.manual_seed(torch.initial_seed() + attempt)

            setup = None
            try:
                # Setup training for this attempt (assembler, optimizer, criterion, etc.)
                setup = self._setup_training_attempt(
                    y, coarse_perms[attempt], hierarchical_group_size,
                    hierarchical_bleed_ratio, hierarchical_bleed_decay,
                    sinkhorn_iters, loss_weight_perm_sparsity,
                    freeze_aggregation_epochs, epochs_per_attempt, annealing_epochs,
                    task_type=task_type,
                    operator_initial_tau=operator_initial_tau,
                    operator_final_tau=operator_final_tau,
                    operator_freeze_min_confidence=operator_freeze_min_confidence,
                    operator_freeze_epochs=operator_freeze_epochs
                )
                
                # Two-phase training: freeze operators for first N epochs (only edges learn)
                if operator_freeze_epochs > 0:
                    if hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'op_logits_per_node'):
                        aggregator = self.assembler.aggregator
                        if aggregator.op_logits_per_node is not None:
                            for logits in aggregator.op_logits_per_node:
                                logits.requires_grad = False
                            # Use "add" as safe default during phase 1 (no gradient explosion from div)
                            if hasattr(aggregator, 'phase1_default_op'):
                                aggregator.phase1_default_op = "add"
                                logging.info(f"   ❄️  Operator selection frozen for first {operator_freeze_epochs} epochs (phase 1: edges only, using 'add')")
                            else:
                                logging.info(f"   ❄️  Operator selection frozen for first {operator_freeze_epochs} epochs (phase 1: edges only)")
                
                # Adaptive temperature annealing parameters
                improvement_patience = 150  # Pause temp decay if no improvement for this many epochs
                improvement_window = 100  # Check improvement over this many epochs (larger = more stable)
                min_improvement_delta = 0.005  # Minimum loss decrease over window (0.5% improvement required)
                confidence_plateau_patience = 1000  # Freeze if confidence doesn't improve for this many epochs after loss convergence

                # Dynamic full-tree egress schedule state (per attempt)
                dynamic_egress_weight = self.loss_weight_full_tree_egress
                egress_baseline_metric = None
                egress_start_epoch = None
                schedule_egress = (
                    self.assembler.tree_layout == "full"
                    and self.loss_weight_full_tree_egress > 0
                    and (full_tree_egress_warmup_epochs > 0 or full_tree_egress_ramp_epochs > 0)
                )
                
                # Initialize R² tracking for R²-modulated regularization
                self._current_r2 = None
                
                for epoch in range(setup.actual_max_epochs):
                    # Curriculum for full-tree concentration:
                    # first learn unconstrained structure, then ramp concentration,
                    # backing off when accuracy degrades.
                    if schedule_egress:
                        latest_metric = setup.accuracy_history[-1] if setup.accuracy_history else None
                        ready_by_metric = latest_metric is not None and latest_metric >= full_tree_egress_start_metric

                        if egress_start_epoch is None and epoch >= full_tree_egress_warmup_epochs and ready_by_metric:
                            egress_start_epoch = epoch

                        if egress_start_epoch is None:
                            dynamic_egress_weight = 0.0
                        else:
                            if egress_baseline_metric is None and setup.accuracy_history:
                                egress_baseline_metric = max(setup.accuracy_history)

                            target_weight = self.loss_weight_full_tree_egress
                            if full_tree_egress_ramp_epochs > 0:
                                ramp_progress = min(
                                    1.0,
                                    (epoch - egress_start_epoch + 1) / max(1, full_tree_egress_ramp_epochs),
                                )
                                planned_weight = target_weight * ramp_progress
                            else:
                                planned_weight = target_weight

                            if egress_baseline_metric is not None and setup.accuracy_history:
                                current_metric = setup.accuracy_history[-1]
                                metric_drop = max(0.0, egress_baseline_metric - current_metric)
                                if metric_drop > full_tree_egress_drop_tolerance:
                                    dynamic_egress_weight = max(
                                        0.0,
                                        dynamic_egress_weight * (1.0 - full_tree_egress_adapt_rate),
                                    )
                                else:
                                    ramp_step = target_weight / max(1, full_tree_egress_ramp_epochs or 100)
                                    dynamic_egress_weight = min(
                                        target_weight,
                                        max(dynamic_egress_weight + ramp_step, planned_weight),
                                    )
                            else:
                                dynamic_egress_weight = planned_weight

                        self._dynamic_full_tree_egress_weight = dynamic_egress_weight
                    else:
                        self._dynamic_full_tree_egress_weight = self.loss_weight_full_tree_egress

                    # Apply training policy (e.g., fixed andness) at start of each epoch
                    if self.training_policy is not None:
                        if hasattr(self.training_policy, 'on_epoch_start'):
                            self.training_policy.on_epoch_start(epoch, setup.actual_max_epochs)
                        self.training_policy.apply(self.assembler)
                    
                    # Unfreeze aggregation after specified epochs
                    if setup.aggregation_frozen and epoch == freeze_aggregation_epochs:
                        for group in setup.param_groups:
                            if group['name'] in ['aggregator', 'other']:
                                for p in group['params']:
                                    p.requires_grad = True
                        setup.aggregation_frozen = False
                        logging.info(f"   🔓 Aggregation unfrozen at epoch {epoch} (joint training begins)")
                    
                    # Unfreeze operator selection after specified epochs (two-phase training)
                    if operator_freeze_epochs > 0 and epoch == operator_freeze_epochs:
                        if hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'op_logits_per_node'):
                            aggregator = self.assembler.aggregator
                            if aggregator.op_logits_per_node is not None:
                                for logits in aggregator.op_logits_per_node:
                                    logits.requires_grad = True
                                # Clear phase1 default so operators can learn
                                if hasattr(aggregator, 'phase1_default_op'):
                                    aggregator.phase1_default_op = None
                                # Reset best tracking - phase 1 states used "add" only, not valid for final model
                                setup.best_overall_loss = float('inf')
                                setup.best_overall_state = None
                                setup.best_overall_epoch = epoch
                                logging.info(f"   🔓 Operator selection unfrozen at epoch {epoch} (phase 2: operators start learning)")
                    
                    self.assembler.train()
                    setup.optimizer.zero_grad(set_to_none=True)
                    just_froze = False  # Flag to skip backward if we freeze during this iteration
                    outputs = self.assembler(x, targets=y)
                    
                    # Compute composite loss with all regularization terms
                    current_sparsity_weight = self.loss_weight_perm_sparsity
                    loss = self._compute_composite_loss(
                        outputs, y, setup.criterion, setup.pos_weight, epoch,
                        sparsity_schedule, setup.use_temperature_annealing, current_sparsity_weight,
                        task_type=setup.task_type
                    )
                    
                    # Track loss and accuracy for smarter plateau detection
                    setup.loss_history.append(loss.item())
                    
                    # Track best overall state (across entire training, not just frozen phase)
                    current_loss_val = loss.item()
                    if current_loss_val < setup.best_overall_loss:
                        setup.best_overall_loss = current_loss_val
                        setup.best_overall_state = {k: v.clone() for k, v in self.assembler.state_dict().items()}
                        setup.best_overall_epoch = epoch
                    
                    # Compute training metric every epoch for plateau detection
                    # This is cheap since we already have outputs
                    with torch.no_grad():
                        if setup.task_type == "regression":
                            # Use R² for regression
                            ss_res = ((y - outputs) ** 2).sum()
                            ss_tot = ((y - y.mean()) ** 2).sum()
                            train_metric = (1 - ss_res / (ss_tot + 1e-8)).item()  # R²
                            # Store R² for use in R²-modulated regularization
                            self._current_r2 = train_metric
                        else:
                            train_predictions = (outputs > 0.5).float()
                            train_metric = (train_predictions == y).float().mean().item()
                            self._current_r2 = None  # Not applicable for classification
                        setup.accuracy_history.append(train_metric)
                    
                    # CHECK CONVERGENCE: If frozen, count epochs. If converged before freeze, wait for freeze.
                    current_loss = loss.item()
                    if self.assembler.is_frozen:
                        # Model is frozen - count epochs since freeze
                        if setup.epoch_when_frozen is None:
                            setup.epoch_when_frozen = epoch
                            setup.best_frozen_loss = current_loss
                            setup.best_frozen_state = {k: v.clone() for k, v in self.assembler.state_dict().items()}
                            
                            # Reduce learning rate significantly for stability during frozen training
                            for param_group in setup.optimizer.param_groups:
                                param_group['lr'] *= 0.01  # 1% of original LR
                            logging.info(f"   🎯 Model frozen at epoch {epoch + 1}, will train for {frozen_training_epochs} more epochs")
                            logging.info(f"      📉 Reduced learning rates to 1% for stable weight refinement")
                        else:
                            # Track best loss and save state during frozen training
                            if current_loss < setup.best_frozen_loss:
                                setup.best_frozen_loss = current_loss
                                setup.best_frozen_state = {k: v.clone() for k, v in self.assembler.state_dict().items()}
                            
                            # Check for divergence: loss > 10x best frozen loss
                            if current_loss > setup.best_frozen_loss * 10 and setup.best_frozen_state is not None:
                                logging.warning(f"   ⚠️  Loss exploded ({current_loss:.4f} > 10x best {setup.best_frozen_loss:.4f}), restoring best frozen state")
                                self.assembler.load_state_dict(setup.best_frozen_state)
                                logging.info(f"   ✅ Restored best frozen state and stopping frozen training")
                                break
                        
                        epochs_since_freeze = epoch - setup.epoch_when_frozen
                        if epochs_since_freeze >= frozen_training_epochs:
                            # Restore best frozen state at end of training
                            if setup.best_frozen_state is not None and current_loss > setup.best_frozen_loss * 1.1:
                                logging.info(f"   📈 Final loss ({current_loss:.4f}) worse than best ({setup.best_frozen_loss:.4f}), restoring best state")
                                self.assembler.load_state_dict(setup.best_frozen_state)
                            logging.info(f"   ✅ Completed {frozen_training_epochs} epochs of frozen training, stopping")
                            break
                    else:
                        # Model not frozen yet - check if converged (waiting to freeze)
                        if current_loss < setup.best_loss_for_convergence - convergence_delta:
                            setup.best_loss_for_convergence = current_loss
                            setup.epochs_without_improvement = 0
                        else:
                            setup.epochs_without_improvement += 1
                        
                        if setup.epochs_without_improvement >= convergence_patience and not setup.has_converged_before_freeze:
                            setup.has_converged_before_freeze = True
                            logging.info(f"   📊 Training converged at epoch {epoch + 1} (no improvement for {convergence_patience} epochs)")
                            logging.info(f"      Loss: {current_loss:.4f}, waiting for permutation confidence > {freeze_confidence_threshold:.2f} to freeze...")
                    
                    # Display epoch progress every 100 epochs
                    if (epoch + 1) % 100 == 0:
                        # Build epoch log message with available temperature info
                        log_parts = [f"   Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}"]
                        
                        # Add permutation info if available
                        if setup.use_temperature_annealing and not self.assembler.is_frozen:
                            perm_temp = self.assembler.input_to_leaf.temperature
                            perm_confidence = 0.0
                            if hasattr(self.assembler.input_to_leaf, 'logits'):
                                with torch.no_grad():
                                    if hasattr(self.assembler.input_to_leaf, 'sinkhorn'):
                                        soft_perm = self.assembler.input_to_leaf.sinkhorn(
                                            self.assembler.input_to_leaf.logits,
                                            temperature=self.assembler.input_to_leaf.temperature,
                                            n_iters=self.assembler.input_to_leaf.sinkhorn_iters
                                        )
                                    else:
                                        soft_perm = torch.softmax(self.assembler.input_to_leaf.logits, dim=1)
                                    perm_confidence = soft_perm.max(dim=1)[0].mean().item()
                            
                            if self.assembler.transformation_layer is not None:
                                trans_temp = self.assembler.transformation_layer.temperature
                                log_parts.append(f"Perm: {perm_temp:.3f}, Trans: {trans_temp:.3f}, Conf: {perm_confidence:.3f}")
                            else:
                                log_parts.append(f"Temp: {perm_temp:.3f}, Conf: {perm_confidence:.3f}")
                        
                        # Add operator tau/confidence (always show if available)
                        if hasattr(self.assembler, 'aggregator'):
                            aggregator = self.assembler.aggregator
                            if hasattr(aggregator, 'tau') and hasattr(aggregator, 'op_logits_per_node') and aggregator.op_logits_per_node is not None:
                                op_tau = aggregator.tau
                                with torch.no_grad():
                                    confidences = []
                                    for logits in aggregator.op_logits_per_node:
                                        probs = torch.softmax(logits / op_tau, dim=0)
                                        confidences.append(probs.max().item())
                                    op_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                                log_parts.append(f"OpTau: {op_tau:.2f}, OpConf: {op_confidence:.2f}")
                        
                        # Show best loss so far (helps track if we're passing good opportunities)
                        if setup.best_overall_loss < float('inf'):
                            log_parts.append(f"Best: {setup.best_overall_loss:.2f}@{setup.best_overall_epoch + 1}")

                        # Show current dynamic full-tree concentration weight when applicable.
                        if self.assembler.tree_layout == "full" and self.loss_weight_full_tree_egress > 0:
                            egr_w = getattr(self, '_dynamic_full_tree_egress_weight', self.loss_weight_full_tree_egress)
                            log_parts.append(f"EgrW: {egr_w:.2f}")
                        
                        logging.info(", ".join(log_parts))
                    
                    # ADAPTIVE TEMPERATURE ANNEALING: Track loss improvement
                    if setup.use_temperature_annealing and not self.assembler.is_frozen:
                        # Track if loss is improving (check over a window for gradual improvements)
                        current_loss = loss.item()
                        
                        # Update best_loss if current is better (for logging)
                        if current_loss < setup.best_loss:
                            setup.best_loss = current_loss
                        
                        # Check improvement over a window (catches gradual progress)
                        if len(setup.loss_history) >= improvement_window:
                            old_loss = setup.loss_history[-improvement_window]
                            loss_improvement = old_loss - current_loss
                            
                            if loss_improvement > min_improvement_delta:
                                # Making progress over the window
                                setup.epochs_since_improvement = 0
                                if setup.temp_paused:
                                    # Only log resume if we're not already at minimum temperature
                                    perm_temp = self.assembler.input_to_leaf.temperature
                                    perm_at_min = perm_temp <= self.permutation_final_temperature * 1.01
                                    if not perm_at_min:
                                        logging.info(f"   ▶️  Resuming temperature annealing (loss improved {loss_improvement:.4f} over {improvement_window} epochs)")
                                    setup.temp_paused = False
                            else:
                                # Not enough improvement over window
                                setup.epochs_since_improvement += 1
                        else:
                            # Not enough history yet, assume making progress
                            setup.epochs_since_improvement = 0
                        
                        # Only anneal temperature if making progress AND within annealing period
                        within_annealing_period = epoch < setup.anneal_over_epochs
                        should_anneal = setup.epochs_since_improvement < improvement_patience and within_annealing_period
                        
                        if should_anneal:
                            # Cool permutation layer
                            self.assembler.input_to_leaf.temperature *= setup.perm_temp_decay_rate
                            
                            # Cool transformation layer independently
                            if self.assembler.transformation_layer is not None and setup.trans_temp_decay_rate is not None:
                                self.assembler.transformation_layer.temperature *= setup.trans_temp_decay_rate
                        else:
                            # Pause temperature annealing - keep exploring at current temperature
                            # Only log if we haven't paused before AND temperatures aren't already at minimum
                            perm_temp = self.assembler.input_to_leaf.temperature
                            perm_at_min = perm_temp <= self.permutation_final_temperature * 1.01  # Within 1% of final
                            
                            if not setup.temp_paused and not perm_at_min:
                                trans_temp = self.assembler.transformation_layer.temperature if self.assembler.transformation_layer is not None else None
                                logging.info(f"   ⏸️  Pausing temperature annealing (no improvement for {improvement_patience} epochs)")
                                trans_temp_str = f"{trans_temp:.3f}" if trans_temp is not None else "N/A"
                                logging.info(f"   🌡️  Frozen temps: Perm={perm_temp:.3f}, Trans={trans_temp_str}")
                                setup.temp_paused = True
                        
                        # Check confidence periodically to catch optimal freezing point
                        should_check_early_freeze = (epoch + 1) % 100 == 0  # Check every 100 epochs
                        temp_ready_to_freeze = self.assembler.input_to_leaf.temperature < self.permutation_final_temperature * 1.1
                        should_check_freeze = temp_ready_to_freeze or should_check_early_freeze
                        
                        if should_check_freeze:
                            # Attempt to freeze permutation if conditions are met
                            just_froze, setup.best_confidence, setup.epochs_without_confidence_improvement, setup.freeze_confidence_warning_shown, skip_frozen_training = \
                                self._check_and_freeze_permutation(
                                    epoch, x_test, y_test, setup.criterion,
                                    freeze_confidence_threshold, freeze_min_confidence,
                                    early_stop_threshold, self.permutation_final_temperature,
                                    setup.has_converged_before_freeze, setup.best_confidence,
                                    setup.epochs_without_confidence_improvement, confidence_plateau_patience,
                                    setup.freeze_confidence_warning_shown, current_loss, binary_threshold,
                                    task_type=setup.task_type,
                                    operator_freeze_min_confidence=operator_freeze_min_confidence,
                                    skip_frozen_threshold=skip_frozen_threshold
                                )
                            
                            # If freeze improved accuracy, stop training immediately
                            if skip_frozen_training:
                                logging.info(f"   🛑 Skipping frozen training, model is optimal at freeze point")
                                break
                    
                    # Anneal operator tau independently - CONTINUES during frozen training
                    # Operators need to commit even after permutation is frozen
                    # Skip annealing during phase 1 (operator_freeze_epochs) - tau stays high for uniform blend
                    if setup.operator_tau_decay_rate is not None and hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'tau'):
                        within_annealing_period = epoch < setup.anneal_over_epochs
                        past_operator_freeze = epoch >= operator_freeze_epochs
                        if within_annealing_period and past_operator_freeze:
                            self.assembler.aggregator.tau *= setup.operator_tau_decay_rate
                    
                    # Anneal full tree temperature independently (works even when permutation is disabled)
                    if self.assembler.tree_layout == "full" and self.assembler.fully_connected_tree is not None:
                        within_annealing_period = epoch < setup.anneal_over_epochs
                        if within_annealing_period and not self.assembler.is_frozen:
                            progress = min(1.0, epoch / setup.anneal_over_epochs)
                            self.assembler.anneal_full_tree_temperature(progress)
                            self.assembler.anneal_full_tree_gumbel(progress)
                    
                    # Anneal alternating tree temperature independently
                    if self.assembler.tree_layout == "alternating" and self.assembler.alternating_tree is not None:
                        within_annealing_period = epoch < setup.anneal_over_epochs
                        if within_annealing_period and not self.assembler.is_frozen:
                            progress = min(1.0, epoch / setup.anneal_over_epochs)
                            self.assembler.anneal_alternating_tree_temperature(progress)
                            self.assembler.anneal_alternating_tree_gumbel(progress)
                    
                    # Perform backward pass and optimizer step (unless we just froze)
                    if not just_froze:
                        loss.backward()
                        # Aggressive gradient clipping to prevent explosion from mul/div operators
                        torch.nn.utils.clip_grad_norm_(self.assembler.parameters(), max_norm=0.5)
                        # Guard against NaN gradients corrupting parameters
                        nan_grad = False
                        for p in self.assembler.parameters():
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                nan_grad = True
                                break
                        if nan_grad:
                            setup.optimizer.zero_grad(set_to_none=True)
                            setup.nan_steps = getattr(setup, 'nan_steps', 0) + 1
                            if setup.nan_steps >= 5:
                                logging.warning(f"   ⚠️  {setup.nan_steps} consecutive NaN gradient steps, aborting attempt")
                                break
                        else:
                            setup.nan_steps = 0
                            setup.optimizer.step()
                                        
                    # Check if transformation layer has converged
                    # Only check transformation convergence when transformation temperature is low
                    # This ensures transformation converges based on its own schedule, not permutation
                    if not setup.transformation_converged and self.assembler.transformation_layer is not None:
                        # Check if transformation temperature is near its final value
                        transformation_is_cool = False
                        if setup.use_temperature_annealing:
                            trans_temp = self.assembler.transformation_layer.temperature
                            # Consider transformation cool when temp is within 2x of final (e.g., < 0.2 for final 0.1)
                            transformation_is_cool = trans_temp < (self.transformation_final_temperature * 2.0)
                        else:
                            # For Hungarian search, wait for permutation to freeze
                            transformation_is_cool = self.assembler.is_frozen
                        
                        if transformation_is_cool:
                            # Use adaptive convergence: 65% confidence for 70% of features
                            if self.assembler.transformation_layer.has_converged(confidence_threshold=0.65, min_converged_ratio=0.7):
                                setup.transformation_converged = True
                                logging.info(f"   ✅ Transformation layer converged at epoch {epoch + 1} (permutation hardened)")

                    # Early stop when frozen/hardened and loss is small
                    early_stop_threshold = self.assembler.early_stop_threshold
                    if self.assembler.transformation_layer is not None and self.assembler.original_input_size >= 10:
                        early_stop_threshold = self.early_stop_threshold_large_inputs  # Configurable threshold for large inputs with transformations
                    
                    if self.assembler.is_frozen and loss.item() < (early_stop_threshold * self.assembler.loss_amplifier):
                        logging.info(f"   ✅ Early stop: loss {loss.item():.4f} < threshold {early_stop_threshold * self.assembler.loss_amplifier:.4f}")
                        break
                    
                    # Also early stop if we achieve perfect training accuracy (no need to continue)
                    # BUT only if temperature is low enough that the solution is stable AND loss is reasonable
                    if len(setup.accuracy_history) > 0 and setup.accuracy_history[-1] >= 0.999:
                        # Check if loss is reasonable (not just lucky predictions with bad model)
                        # With loss_amplifier, acceptable loss threshold is early_stop_threshold * loss_amplifier
                        loss_reasonable = loss.item() < (early_stop_threshold * self.assembler.loss_amplifier * 2.0)  # 2x margin for perfect accuracy
                        
                        if loss_reasonable:
                            # Check if permutation is stable AND confident (temperature low or frozen)
                            if setup.use_temperature_annealing and not self.assembler.is_frozen:
                                current_perm_temp = self.assembler.input_to_leaf.temperature
                                permutation_stable = current_perm_temp < 1.0  # Temperature cool enough for stable solution
                                
                                # Also check if permutation is confident (peaked distributions)
                                # Don't freeze if permutation is flat/uniform even with low temperature
                                if permutation_stable and hasattr(self.assembler.input_to_leaf, 'logits'):
                                    perm_probs = torch.softmax(self.assembler.input_to_leaf.logits / current_perm_temp, dim=1)
                                    confidence = perm_probs.max(dim=1)[0].mean().item()
                                    permutation_confident = confidence > 0.8  # Require confident permutation
                                    permutation_stable = permutation_stable and permutation_confident
                            else:
                                permutation_stable = self.assembler.is_frozen
                            
                            if permutation_stable:
                                logging.info(f"   ✅ Early stop: perfect training accuracy (100.0%) with stable permutation and loss {loss.item():.2f}")
                                # Freeze the model to mark it as ready
                                self.assembler.is_frozen = True
                                break
                            # else: accuracy is 100% but temperature still high or not confident - keep training
                        # else: accuracy is high but loss is too high - likely unstable, keep training

                # Restore best overall state if current is significantly worse
                # BUT only if model structure hasn't changed (i.e., we haven't frozen yet)
                # After freezing, input_to_leaf changes from logits to P_hard, causing state_dict mismatch
                if setup.best_overall_state is not None:
                    final_loss = setup.loss_history[-1] if setup.loss_history else float('inf')
                    if final_loss > setup.best_overall_loss * 1.5:  # Final loss > 1.5x best
                        logging.info(f"   📈 Final loss ({final_loss:.4f}) worse than best ({setup.best_overall_loss:.4f} at epoch {setup.best_overall_epoch + 1})")
                        
                        # Check if model structure matches saved state
                        current_keys = set(self.assembler.state_dict().keys())
                        saved_keys = set(setup.best_overall_state.keys())
                        if current_keys == saved_keys:
                            logging.info(f"      Restoring best overall state...")
                            self.assembler.load_state_dict(setup.best_overall_state)
                        else:
                            # Model structure changed (e.g., frozen) - can't restore pre-freeze state
                            # Use best_frozen_state instead if available
                            if setup.best_frozen_state is not None:
                                logging.info(f"      Model structure changed (frozen). Using best frozen state instead.")
                                self.assembler.load_state_dict(setup.best_frozen_state)
                            else:
                                logging.info(f"      Model structure changed (frozen). Cannot restore pre-freeze state.")
                
                # Force freeze if not actually frozen (check locked_perm, not just flag)
                # Early stopping may set is_frozen=True without creating locked_perm
                actually_frozen = self.assembler.locked_perm is not None
                if not actually_frozen and force_freeze:
                    logging.info(f"   📋 Model structure not frozen (no locked_perm), checking if force-freeze is possible...")
                    if hasattr(self.assembler.input_to_leaf, 'logits'):
                        try:
                            logging.info(f"   🔒 Force-freezing model (creating locked_perm and frozen structure)")
                            from bacon.frozonInputToLeaf import frozenInputToLeaf
                            from scipy.optimize import linear_sum_assignment
                            import numpy as np
                            
                            # Get hard assignment from current soft permutation
                            soft_perm = torch.softmax(self.assembler.input_to_leaf.logits, dim=1)
                            
                            # BUG FIX: Use Hungarian algorithm instead of argmax to ensure valid permutation
                            # argmax can create duplicates when multiple rows have max in same column
                            soft_perm_np = soft_perm.detach().cpu().numpy()
                            row_ind, col_ind = linear_sum_assignment(-soft_perm_np)
                            hard_perm = torch.tensor(col_ind[row_ind.argsort()], dtype=torch.long, device=self.assembler.device)
                            
                            self.assembler.locked_perm = hard_perm.clone().detach()
                            self.assembler.is_frozen = True
                            # Replace with frozen layer
                            self.assembler.input_to_leaf = frozenInputToLeaf(
                                self.assembler.locked_perm,
                                self.assembler.original_input_size
                            ).to(self.assembler.device)
                            logging.info(f"   ✅ Successfully force-frozen model with valid permutation (no duplicates)")
                        except Exception as e:
                            logging.warning(f"   ⚠️ Failed to force-freeze model: {e}")
                    else:
                        # No permutation layer to freeze (e.g., Identity when use_permutation_layer=False)
                        # For full-tree mode, harden the learned discrete structure before scoring.
                        if self.assembler.tree_layout == "full" and self.assembler.fully_connected_tree is not None:
                            self.assembler.harden_full_tree(mode="auto")
                            if hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'harden_operators'):
                                self.assembler.aggregator.harden_operators()
                            logging.info(f"   ✅ No permutation layer to freeze; hardened full-tree structure for evaluation")
                        elif self.assembler.tree_layout == "alternating" and self.assembler.alternating_tree is not None:
                            self.assembler.harden_alternating_tree()
                            if hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'harden_operators'):
                                self.assembler.aggregator.harden_operators()
                            logging.info(f"   ✅ No permutation layer to freeze; hardened alternating-tree structure for evaluation")
                        else:
                            # Just mark as frozen since there's nothing to permute
                            self.assembler.is_frozen = True
                            logging.info(f"   ✅ No permutation layer to freeze (already identity), marking as frozen")
                else:
                    # Check if actually frozen (not just flag set)
                    actually_frozen = not hasattr(self.assembler.input_to_leaf, 'logits')
                    if actually_frozen:
                        logging.info(f"   ✅ Model already frozen naturally during training")
                    else:
                        logging.info(f"   ℹ️  Model has is_frozen=True but still has soft permutation (not force-frozen)")

                # Always evaluate each attempt, regardless of freeze status
                # This ensures we track the best permutation even if it didn't fully freeze
                if task_type == "regression":
                    output = self.inference_raw(x_test)
                    ss_res = ((y_test - output) ** 2).sum()
                    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                    metric = (1 - ss_res / (ss_tot + 1e-8)).item()  # R²
                    metric_name = "R²"
                elif binary_threshold >= 0:
                    metric = self.evaluate(x_test, y_test, threshold=binary_threshold)
                    metric_name = "accuracy"
                else:
                    output = self.inference_raw(x_test)
                    mae = (output - y_test).abs().mean().item()
                    metric = 1.0 - min(mae, 1.0)
                    metric_name = "accuracy"
                # Check actual frozen status, not just the flag
                actually_frozen = not hasattr(self.assembler.input_to_leaf, 'logits')
                frozen_status = "frozen" if actually_frozen else "unfrozen"
                logging.info(f"✅ Attempt {attempt + 1} {metric_name}: {metric:.4f} ({frozen_status})")
                
                if metric > best_metric:
                    best_metric = metric
                    best_model = {k: v.clone() for k, v in self.assembler.state_dict().items()}
                    best_is_frozen = self.assembler.is_frozen
                    best_locked_perm = self.assembler.locked_perm.clone() if self.assembler.locked_perm is not None else None
                    best_full_tree_hardened = (
                        self.assembler.tree_layout == "full"
                        and self.assembler.fully_connected_tree is not None
                        and self.assembler.fully_connected_tree.hardened_edges is not None
                    )
                    best_alternating_hardened = (
                        self.assembler.tree_layout == "alternating"
                        and self.assembler.alternating_tree is not None
                        and all(
                            not getattr(layer, 'learn_routing', False) or getattr(layer, 'hard_edges', None) is not None
                            for layer in self.assembler.alternating_tree.agg_layers
                        )
                    )
                    best_operator_hardened = hasattr(self.assembler, 'aggregator') and getattr(self.assembler.aggregator, 'force_hard_selection', False)
                    logging.info(f"   🏆 New best model! {metric_name}: {best_metric:.4f}")
                    
                    # Save intermediate best model after each improvement (only if better than loaded model)
                    if save_model and save_path and best_metric > loaded_metric:
                        # Temporarily store current state before loading best
                        temp_state = {k: v.clone() for k, v in self.assembler.state_dict().items()}
                        temp_is_frozen = self.assembler.is_frozen
                        temp_locked_perm = self.assembler.locked_perm.clone() if self.assembler.locked_perm is not None else None
                        temp_input_layer = self.assembler.input_to_leaf
                        
                        # Load best model and restore its frozen state
                        self.assembler.load_state_dict(best_model)
                        if best_is_frozen and best_locked_perm is not None:
                            self.assembler.is_frozen = True
                            self.assembler.locked_perm = best_locked_perm
                            # Recreate frozen input layer
                            from bacon.frozonInputToLeaf import frozenInputToLeaf
                            self.assembler.input_to_leaf = frozenInputToLeaf(
                                best_locked_perm,
                                self.assembler.original_input_size
                            ).to(self.assembler.device)
                            # CRITICAL: Load the weights into the frozen layer
                            self.assembler.load_state_dict(best_model)
                        elif best_full_tree_hardened:
                            self.assembler.harden_full_tree(mode="auto")
                            if best_operator_hardened and hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'harden_operators'):
                                self.assembler.aggregator.harden_operators()
                        elif best_alternating_hardened:
                            self.assembler.harden_alternating_tree()
                            if best_operator_hardened and hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'harden_operators'):
                                self.assembler.aggregator.harden_operators()
                        
                        logging.info(f"   💾 Saving intermediate best model to {save_path}")
                        self.save_model(save_path)
                        
                        # Restore current training state completely
                        self.assembler.load_state_dict(temp_state)
                        self.assembler.is_frozen = temp_is_frozen
                        self.assembler.locked_perm = temp_locked_perm
                        self.assembler.input_to_leaf = temp_input_layer
                    
                    if best_metric >= acceptance_threshold:
                        logging.info(f"   ✅ Acceptance threshold reached ({acceptance_threshold:.4f})")
                        break
                        
            except RuntimeError as e:
                logging.error(f"🔥 Attempt {attempt + 1} failed with error: {e}")
            except Exception as e:
                logging.error(f"🔥 Attempt {attempt + 1} failed with error: {e}")
            finally:
                # Restore original sparsity weight if it was overridden
                if setup is not None and setup.original_sparsity_weight is not None:
                    self.loss_weight_perm_sparsity = setup.original_sparsity_weight
                if hasattr(self, '_dynamic_full_tree_egress_weight'):
                    delattr(self, '_dynamic_full_tree_egress_weight')
        
        if best_model is None:
            raise ValueError("No model met the acceptance threshold.")
        
        # Determine metric name for logging
        metric_name = "R²" if task_type == "regression" else "accuracy"
        
        # For regression, loaded_metric starts at -inf, so any real value is better
        # For classification, loaded_metric starts at 0
        has_loaded_model = loaded_metric > (-float('inf') if task_type == "regression" else 0)
        
        # If new model is worse than loaded model, reload the original from disk
        if has_loaded_model and best_metric <= loaded_metric and save_path and os.path.exists(save_path):
            logging.info(f"⚠️ New model {metric_name} {best_metric:.4f} <= loaded model {metric_name} {loaded_metric:.4f}")
            logging.info(f"   🔄 Reloading original model from {save_path}")
            self.load_model(save_path)
            return self.assembler.state_dict(), loaded_metric
        
        # Load best model and restore its frozen state
        self.assembler.load_state_dict(best_model)
        if best_is_frozen and best_locked_perm is not None:
            self.assembler.is_frozen = True
            self.assembler.locked_perm = best_locked_perm
            # Recreate frozen input layer
            from bacon.frozonInputToLeaf import frozenInputToLeaf
            self.assembler.input_to_leaf = frozenInputToLeaf(
                best_locked_perm,
                self.assembler.original_input_size
            ).to(self.assembler.device)
        elif best_full_tree_hardened:
            self.assembler.harden_full_tree(mode="auto")
            if best_operator_hardened and hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'harden_operators'):
                self.assembler.aggregator.harden_operators()
        elif best_alternating_hardened:
            self.assembler.harden_alternating_tree()
            if best_operator_hardened and hasattr(self.assembler, 'aggregator') and hasattr(self.assembler.aggregator, 'harden_operators'):
                self.assembler.aggregator.harden_operators()
        
        if save_model:
            if not has_loaded_model or best_metric > loaded_metric:
                logging.info(f"✅ Saving the best model with {metric_name} {best_metric:.4f} to {save_path}")
                self.save_model(save_path)
            else:
                logging.info(f"⚠️ New model {metric_name} {best_metric:.4f} <= loaded model {metric_name} {loaded_metric:.4f}, not overwriting {save_path}")
        return best_model, best_metric
    
    def prune_features(self, features):        
        return self.assembler.prune_features(features=features)    