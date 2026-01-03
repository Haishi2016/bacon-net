import torch.nn as nn
import torch
from bacon.binaryTreeLogicNet import binaryTreeLogicNet
import logging
import os
from dataclasses import dataclass
from typing import Optional
from bacon.aggregators.lsp import FullWeightAggregator, HalfWeightAggregator
from bacon.aggregators.bool import MinMaxAggregator

_aggregator_registry = {
    "lsp.full_weight": FullWeightAggregator,
    "lsp.half_weight": HalfWeightAggregator,
    "bool.min_max": MinMaxAggregator
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
    anneal_over_epochs: int
    original_sparsity_weight: Optional[float]
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
        lr_permutation (float, optional): Learning rate for permutation layer. Defaults to 0.3. Higher = faster exploration of feature orderings.
        lr_transformation (float, optional): Learning rate for transformation layer. Defaults to 0.5. Higher = faster transformation selection.
        lr_aggregator (float, optional): Learning rate for aggregator weights. Defaults to 0.1. Lower = more stable tree structure.
        lr_other (float, optional): Learning rate for other parameters. Defaults to 0.1.
        use_class_weighting (bool, optional): Whether to apply class weighting for imbalanced data. Defaults to True. When True, penalizes minority class errors more heavily (pos_weight = neg_count/pos_count). When False, uses standard BCE loss (original behavior).
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
                 lr_permutation=0.3,
                 lr_transformation=0.5,
                 lr_aggregator=0.1,
                 lr_other=0.1,
                 use_class_weighting=True):
        super(baconNet, self).__init__()        
        if aggregator not in _aggregator_registry:
            raise ValueError(f"Unknown aggregator: {aggregator}. Available options: {list(_aggregator_registry.keys())}")
        aggregator_class = _aggregator_registry[aggregator]
        aggregator = aggregator_class()
        self.is_frozen = is_frozen
        self.early_stop_threshold_large_inputs = early_stop_threshold_large_inputs
        self.permutation_initial_temperature = permutation_initial_temperature
        self.permutation_final_temperature = permutation_final_temperature
        self.transformation_initial_temperature = transformation_initial_temperature
        self.transformation_final_temperature = transformation_final_temperature
        
        # Use transformation_initial_temperature if transformation_temperature not specified
        if transformation_temperature is None:
            transformation_temperature = transformation_initial_temperature
        
        # Loss component weights (normalized internally during training)
        self.loss_weight_main = loss_weight_main  # Main BCE loss
        self.loss_weight_perm_entropy = loss_weight_perm_entropy  # Permutation entropy regularization
        self.loss_weight_trans_entropy = loss_weight_trans_entropy  # Transformation entropy regularization
        self.loss_weight_perm_sparsity = loss_weight_perm_sparsity  # Permutation sparsity loss (encourage peaked distributions)
        
        # Learning rates for different parameter groups
        self.lr_permutation = lr_permutation
        self.lr_transformation = lr_transformation
        self.lr_aggregator = lr_aggregator
        self.lr_other = lr_other
        
        # Class weighting for imbalanced data
        self.use_class_weighting = use_class_weighting
        
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
                                            weight_choices=None)
        
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
            predictions = (outputs > threshold).float()
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
            predictions = (outputs > threshold).float()  # Binarize the output to match the target labels (0 or 1)
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
                                freeze_aggregation_epochs, epochs_per_attempt, annealing_epochs):
        """Set up all components needed for a training attempt.
        
        Returns:
            TrainingSetup: Dataclass containing all setup state
        """
        # Re-initialize assembler
        cfg = self.assembler
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
            aggregator=cfg.aggregator,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            early_stop_threshold=cfg.early_stop_threshold,
            use_transformation_layer=cfg.use_transformation_layer,
            transformation_temperature=self.transformation_initial_temperature,
            transformation_use_gumbel=cfg.transformation_layer.use_gumbel if cfg.transformation_layer else False,
            transformations=cfg._custom_transformations if hasattr(cfg, '_custom_transformations') else None,
            device=cfg.device,
            sinkhorn_iters=sinkhorn_iters,
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
        
        # Setup criterion with class weighting
        if self.use_class_weighting:
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
                pos_weight = None
        else:
            logging.info(f"   ⚖️  Class weighting disabled: using standard BCE loss (original behavior)")
            criterion = nn.BCELoss()
            pos_weight = None
        
        # Handle sparsity weight override
        original_sparsity_weight = None
        if loss_weight_perm_sparsity is not None:
            original_sparsity_weight = self.loss_weight_perm_sparsity
            self.loss_weight_perm_sparsity = loss_weight_perm_sparsity
            logging.info(f"   🎯 Using custom sparsity loss weight: {loss_weight_perm_sparsity}")
        
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
            anneal_over_epochs=anneal_over_epochs_val,
            original_sparsity_weight=original_sparsity_weight,
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
            transformation_converged=False
        )
    
    def _compute_composite_loss(self, outputs, y, criterion, pos_weight, epoch, 
                                sparsity_schedule, use_temperature_annealing, current_sparsity_weight):
        """Compute composite loss with all regularization terms."""
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            logging.error(f"   ❌ NaN detected in model outputs at epoch {epoch + 1}")
            logging.error(f"      Output stats: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, mean={outputs.mean().item():.6f}")
            outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        # Compute main BCE loss with optional class weighting
        if self.use_class_weighting and pos_weight is not None:
            bce_losses = criterion(outputs, y)
            if torch.isnan(bce_losses).any():
                logging.error(f"   ❌ NaN in BCE losses (weighted)")
            weights = torch.where(y == 1, pos_weight, 1.0)
            main_loss = (bce_losses * weights).mean() * self.assembler.loss_amplifier
        else:
            main_loss = criterion(outputs, y) * self.assembler.loss_amplifier
        
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
        
        # Weight regularization
        if self.assembler.weight_mode == "trainable":
            depth_weight_penalty = 0.0
            for w in self.assembler.weights:
                depth_weight_penalty += self.assembler.weight_penalty_strength * ((torch.sigmoid(w) - 0.5) ** 2).mean()
            loss = loss + depth_weight_penalty
        
        return loss
    
    def _check_and_freeze_permutation(self, epoch, x_test, y_test, criterion, 
                                     freeze_confidence_threshold, freeze_min_confidence,
                                     early_stop_threshold, perm_final_temp,
                                     has_converged_before_freeze, best_confidence,
                                     epochs_without_confidence_improvement, confidence_plateau_patience,
                                     freeze_confidence_warning_shown, current_loss):
        """Check if permutation should be frozen and perform freezing if ready.
        
        Returns:
            tuple: (just_froze, best_confidence, epochs_without_confidence_improvement, freeze_confidence_warning_shown)
        """
        just_froze = False
        
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
                
                # Freeze conditions
                confidence_reached = mean_confidence >= freeze_confidence_threshold
                loss_threshold = early_stop_threshold * self.assembler.loss_amplifier * 1.5
                good_confidence_and_loss = (mean_confidence >= freeze_min_confidence and current_loss < loss_threshold)
                confidence_plateaued = (has_converged_before_freeze and 
                                       epochs_without_confidence_improvement >= confidence_plateau_patience)
                
                should_freeze = confidence_reached or good_confidence_and_loss or confidence_plateaued
                
                if not should_freeze:
                    temp_ready_to_freeze = self.assembler.input_to_leaf.temperature < perm_final_temp * 1.1
                    if temp_ready_to_freeze and not freeze_confidence_warning_shown:
                        logging.info(f"   ⚠️  Permutation temp low ({self.assembler.input_to_leaf.temperature:.3f}) but not ready to freeze")
                        logging.info(f"      Mean max probability: {mean_confidence:.3f} (target: {freeze_confidence_threshold:.2f})")
                        if has_converged_before_freeze:
                            logging.info(f"      Loss converged, waiting for confidence plateau ({epochs_without_confidence_improvement}/{confidence_plateau_patience} epochs)")
                        else:
                            logging.info(f"      Waiting for loss convergence first...")
                        freeze_confidence_warning_shown = True
                    just_froze = True
                    return just_froze, best_confidence, epochs_without_confidence_improvement, freeze_confidence_warning_shown
                
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
                    before_pred = (before_output > 0.5).float()
                    before_accuracy = (before_pred == y_test).float().mean().item()
                    
                    max_probs = soft_perm.max(dim=1)[0]
                    logging.info(f"      📊 BEFORE FREEZE (test set):")
                    logging.info(f"         Loss: {before_loss.item():.4f}, Test Accuracy: {before_accuracy:.4f}")
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
                
                # Use Hungarian algorithm
                soft_perm_np = soft_perm.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-soft_perm_np)
                hard_perm = torch.tensor(col_ind[row_ind.argsort()], dtype=torch.long, device=self.assembler.device)
                logging.info(f"      Using Hungarian algorithm for optimal valid permutation")
                
                perm_differences = (argmax_perm != hard_perm).sum().item()
                if perm_differences > 0:
                    logging.info(f"      ⚠️  Hungarian differs from argmax in {perm_differences}/{len(hard_perm)} positions")
                
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
                    after_pred = (after_output > 0.5).float()
                    after_accuracy = (after_pred == y_test).float().mean().item()
                    
                    logging.info(f"      📊 AFTER FREEZE:")
                    logging.info(f"         Loss: {after_loss.item():.4f} (Δ={after_loss.item()-before_loss.item():+.4f})")
                    logging.info(f"         Test Accuracy: {after_accuracy:.4f} (Δ={after_accuracy-before_accuracy:+.4f})")
                    
                    pred_changes = (before_pred != after_pred).sum().item()
                    if pred_changes > 0:
                        logging.info(f"         ⚠️  {pred_changes}/{len(y_test)} predictions changed after freeze")
                    
                    output_diff = (after_output - before_output).abs().max().item()
                    logging.info(f"         Max output difference: {output_diff:.6f}")
                
                self.assembler.train()
                logging.info(f"   🔒 Successfully frozen model (locked_perm created)")
                just_froze = True
                
            except Exception as e:
                logging.warning(f"   ⚠️ Failed to freeze on hardening: {e}")
                self.assembler.is_frozen = True
                just_froze = True
        
        return just_froze, best_confidence, epochs_without_confidence_improvement, freeze_confidence_warning_shown

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
                        freeze_min_confidence = 0.85,  # Minimum confidence for early freeze with low loss (raised from 0.75)
                        loss_weight_perm_sparsity = None,
                        sparsity_schedule = None,  # (initial_weight, final_weight, transition_epochs)
                        freeze_aggregation_epochs = 0,  # Freeze aggregation for first N epochs
                        save_model = True,
                        use_hierarchical_permutation = False,
                        force_freeze = True,
                        hierarchical_group_size = 3,
                        hierarchical_epochs_per_attempt = None,
                        hierarchical_bleed_ratio = 0.1,
                        hierarchical_bleed_decay = 2.0,
                        sinkhorn_iters = 100,
                        binary_threshold=0.5):
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
        Returns:
            tuple: Best model state dictionary and its accuracy.
        """
        best_accuracy = 0.0
        best_model = None        

        if os.path.exists(save_path):
            try:
                logging.info(f"📂 Found saved model at {save_path}, loading...")
                self.load_model(save_path)
                if self.assembler.is_frozen:
                    if binary_threshold >= 0:
                        acc = self.evaluate(x_test, y_test, threshold=binary_threshold)
                    else:
                        output = self.inference_raw(x_test)
                        mae = (output - y_test).abs().mean().item()
                        acc = 1.0 - min(mae, 1.0)
                    logging.info(f"✅ Loaded model accuracy: {acc:.4f}")
                    if acc >= acceptance_threshold:
                        return self.assembler.state_dict(), acc
                    else:
                        logging.info(f"⚠️ Loaded accuracy {acc:.4f} < threshold {acceptance_threshold:.4f}, will retrain")
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

        for attempt in range(total_attempts):
            if use_hierarchical_permutation and coarse_perms[attempt] is not None:
                logging.info(f"🔥 Attempting coarse permutation {attempt + 1}/{total_attempts}: {coarse_perms[attempt]}")
            else:
                logging.info(f"🔥 Attempting to find the best model... {attempt + 1}/{total_attempts}")

            torch.manual_seed(torch.initial_seed() + attempt)

            try:
                # Setup training for this attempt (assembler, optimizer, criterion, etc.)
                setup = self._setup_training_attempt(
                    y, coarse_perms[attempt], hierarchical_group_size,
                    hierarchical_bleed_ratio, hierarchical_bleed_decay,
                    sinkhorn_iters, loss_weight_perm_sparsity,
                    freeze_aggregation_epochs, epochs_per_attempt, annealing_epochs
                )
                
                # Adaptive temperature annealing parameters
                improvement_patience = 150  # Pause temp decay if no improvement for this many epochs
                improvement_window = 100  # Check improvement over this many epochs (larger = more stable)
                min_improvement_delta = 0.005  # Minimum loss decrease over window (0.5% improvement required)
                confidence_plateau_patience = 1000  # Freeze if confidence doesn't improve for this many epochs after loss convergence
                
                for epoch in range(setup.actual_max_epochs):
                    # Unfreeze aggregation after specified epochs
                    if setup.aggregation_frozen and epoch == freeze_aggregation_epochs:
                        for group in setup.param_groups:
                            if group['name'] in ['aggregator', 'other']:
                                for p in group['params']:
                                    p.requires_grad = True
                        setup.aggregation_frozen = False
                        logging.info(f"   🔓 Aggregation unfrozen at epoch {epoch} (joint training begins)")
                    
                    self.assembler.train()
                    setup.optimizer.zero_grad(set_to_none=True)
                    just_froze = False  # Flag to skip backward if we freeze during this iteration
                    outputs = self.assembler(x, targets=y)
                    
                    # Compute composite loss with all regularization terms
                    current_sparsity_weight = self.loss_weight_perm_sparsity
                    loss = self._compute_composite_loss(
                        outputs, y, setup.criterion, setup.pos_weight, epoch,
                        sparsity_schedule, setup.use_temperature_annealing, current_sparsity_weight
                    )
                    
                    # Track loss and accuracy for smarter plateau detection
                    setup.loss_history.append(loss.item())
                    
                    # Compute training accuracy every epoch for plateau detection
                    # This is cheap since we already have outputs
                    with torch.no_grad():
                        train_predictions = (outputs > 0.5).float()
                        train_accuracy = (train_predictions == y).float().mean().item()
                        setup.accuracy_history.append(train_accuracy)
                    
                    # CHECK CONVERGENCE: If frozen, count epochs. If converged before freeze, wait for freeze.
                    current_loss = loss.item()
                    if self.assembler.is_frozen:
                        # Model is frozen - count epochs since freeze
                        if setup.epoch_when_frozen is None:
                            setup.epoch_when_frozen = epoch
                            logging.info(f"   🎯 Model frozen at epoch {epoch + 1}, will train for {frozen_training_epochs} more epochs")
                        
                        epochs_since_freeze = epoch - setup.epoch_when_frozen
                        if epochs_since_freeze >= frozen_training_epochs:
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
                        if setup.use_temperature_annealing and not self.assembler.is_frozen:
                            perm_temp = self.assembler.input_to_leaf.temperature
                            
                            # Calculate permutation confidence for monitoring convergence
                            perm_confidence = 0.0
                            if hasattr(self.assembler.input_to_leaf, 'logits'):
                                with torch.no_grad():
                                    # Use Sinkhorn normalization to get actual doubly stochastic matrix
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
                                logging.info(f"   Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}, Perm: {perm_temp:.3f}, Trans: {trans_temp:.3f}, Conf: {perm_confidence:.3f}")
                            else:
                                logging.info(f"   Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}, Temp: {perm_temp:.3f}, Conf: {perm_confidence:.3f}")
                        else:
                            logging.info(f"   Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}")
                    
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
                            just_froze, setup.best_confidence, setup.epochs_without_confidence_improvement, setup.freeze_confidence_warning_shown = \
                                self._check_and_freeze_permutation(
                                    epoch, x_test, y_test, setup.criterion,
                                    freeze_confidence_threshold, freeze_min_confidence,
                                    early_stop_threshold, self.permutation_final_temperature,
                                    setup.has_converged_before_freeze, setup.best_confidence,
                                    setup.epochs_without_confidence_improvement, confidence_plateau_patience,
                                    setup.freeze_confidence_warning_shown, current_loss
                                )
                    
                    # Perform backward pass and optimizer step (unless we just froze)
                    if not just_froze:
                        loss.backward()
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
                        logging.warning(f"   ⚠️ Cannot force-freeze: input_to_leaf has no 'logits' attribute")
                else:
                    # Check if actually frozen (not just flag set)
                    actually_frozen = not hasattr(self.assembler.input_to_leaf, 'logits')
                    if actually_frozen:
                        logging.info(f"   ✅ Model already frozen naturally during training")
                    else:
                        logging.info(f"   ℹ️  Model has is_frozen=True but still has soft permutation (not force-frozen)")

                # Always evaluate each attempt, regardless of freeze status
                # This ensures we track the best permutation even if it didn't fully freeze
                if binary_threshold >= 0:
                    accuracy = self.evaluate(x_test, y_test, threshold=binary_threshold)
                else:
                    output = self.inference_raw(x_test)
                    mae = (output - y_test).abs().mean().item()
                    accuracy = 1.0 - min(mae, 1.0)
                # Check actual frozen status, not just the flag
                actually_frozen = not hasattr(self.assembler.input_to_leaf, 'logits')
                frozen_status = "frozen" if actually_frozen else "unfrozen"
                logging.info(f"✅ Attempt {attempt + 1} accuracy: {accuracy:.4f} ({frozen_status})")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = self.assembler.state_dict()
                    best_is_frozen = self.assembler.is_frozen
                    best_locked_perm = self.assembler.locked_perm.clone() if self.assembler.locked_perm is not None else None
                    logging.info(f"   🏆 New best model! Accuracy: {best_accuracy:.4f}")
                    
                    # Save intermediate best model after each improvement
                    if save_model and save_path:
                        # Temporarily store current state before loading best
                        temp_state = self.assembler.state_dict()
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
                        
                        logging.info(f"   💾 Saving intermediate best model to {save_path}")
                        self.save_model(save_path)
                        
                        # Restore current training state completely
                        self.assembler.load_state_dict(temp_state)
                        self.assembler.is_frozen = temp_is_frozen
                        self.assembler.locked_perm = temp_locked_perm
                        self.assembler.input_to_leaf = temp_input_layer
                    
                    if best_accuracy >= acceptance_threshold:
                        logging.info(f"   ✅ Acceptance threshold reached ({acceptance_threshold:.4f})")
                        break
                        
            except RuntimeError as e:
                logging.error(f"🔥 Attempt {attempt + 1} failed with error: {e}")
            finally:
                # Restore original sparsity weight if it was overridden
                if setup.original_sparsity_weight is not None:
                    self.loss_weight_perm_sparsity = setup.original_sparsity_weight
        
        if best_model is None:
            raise ValueError("No model met the acceptance threshold.")
        
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
        
        if save_model:
            logging.info(f"✅ Saving the best model with accuracy {best_accuracy:.4f} to {save_path}")
            self.save_model(save_path)
        return best_model, best_accuracy
    
    def prune_features(self, features):        
        return self.assembler.prune_features(features=features)    