import torch.nn as nn
import torch
from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.frozonInputToLeaf import frozenInputToLeaf
from bacon.aggregators.lsp import FullWeightAggregator, HalfWeightAggregator
from bacon.aggregators.bool import MinMaxAggregator
import logging

_aggregator_registry = {
    "lsp.full_weight": FullWeightAggregator,
    "lsp.half_weight": HalfWeightAggregator,
    "bool.min_max": MinMaxAggregator
}

class HierarchicalBaconNet(nn.Module):
    """
    Hierarchical BACON network that handles grouped features with two-level structure:
    - Sub-trees: Process features within each group (frozen order, transformations + aggregation)
    - Global tree: Aggregates transformed group outputs (permutation + group transformations + aggregation)
    
    Args:
        feature_groups (dict): Dictionary mapping group names to lists of feature indices.
                              Example: {'payment_history': [0,1,2,3,4,5,6], 'demographics': [7,8,9,10]}
        total_features (int): Total number of input features across all groups.
        freeze_loss_threshold (float, optional): Loss threshold for freezing global permutation. Defaults to 0.25.
                                                Note: If using loss_amplifier, this will be multiplied by it.
        weight_mode (str, optional): Weight mode for trees ('fixed' or 'trainable'). Defaults to 'fixed'.
        aggregator (str, optional): Aggregator type. Defaults to 'lsp.half_weight'.
        use_sub_tree_transformation (bool, optional): Enable transformations in sub-trees. Defaults to True.
        use_global_transformation (bool, optional): Enable transformations in global tree. Defaults to True.
        sub_tree_layout (str, optional): Layout for sub-trees. Defaults to 'left'.
        global_tree_layout (str, optional): Layout for global tree. Defaults to 'balanced'.
        max_permutations (int, optional): Max permutations for global tree. Defaults to 10000.
        loss_amplifier (float, optional): Amplifier for the loss. Defaults to 1.0.
        lr_permutation (float, optional): Learning rate for global permutation. Defaults to 0.3.
        lr_transformation (float, optional): Learning rate for transformations. Defaults to 0.5.
        lr_aggregator (float, optional): Learning rate for aggregator weights. Defaults to 0.1.
        lr_other (float, optional): Learning rate for other parameters. Defaults to 0.1.
        use_class_weighting (bool, optional): Enable class weighting for imbalanced data. Defaults to True.
        sub_tree_transformations (list, optional): List of transformation objects for sub-trees. 
                                                   If None, uses all transformations. 
                                                   Example: [IdentityTransformation(n), NegationTransformation(n)]
        global_tree_transformations (list, optional): List of transformation objects for global tree.
                                                      If None, uses all transformations.
        sub_tree_permutation_groups (list, optional): List of group names that should learn permutations.
                                                      If None or empty, all sub-trees use identity permutation.
                                                      Example: ['demographics'] to enable permutation only on demographics group.
        permutation_initial_temperature (float, optional): Starting temperature for permutation annealing. Defaults to 5.0.
        permutation_final_temperature (float, optional): Final temperature for permutation annealing. Defaults to 0.1.
        transformation_initial_temperature (float, optional): Starting temperature for transformation selection. Defaults to 1.0.
                                                              Lower than permutation since transformation has 2^n vs n! states.
        transformation_final_temperature (float, optional): Final temperature for transformation selection. Defaults to 0.1.
    """
    def __init__(self, 
                 feature_groups,
                 total_features,
                 freeze_loss_threshold=0.25,
                 weight_mode='fixed',
                 aggregator='lsp.half_weight',
                 use_sub_tree_transformation=True,
                 use_global_transformation=True,
                 sub_tree_layout='left',
                 global_tree_layout='balanced',
                 max_permutations=10000,
                 loss_amplifier=1.0,
                 lr_permutation=0.3,
                 lr_transformation=0.5,
                 lr_aggregator=0.1,
                 lr_other=0.1,
                 use_class_weighting=True,
                 sub_tree_transformations=None,
                 global_tree_transformations=None,
                 sub_tree_permutation_groups=None,
                 permutation_initial_temperature=5.0,
                 permutation_final_temperature=0.1,
                 transformation_initial_temperature=1.0,
                 transformation_final_temperature=0.1):
        super(HierarchicalBaconNet, self).__init__()
        
        # Resolve aggregator string to object
        if isinstance(aggregator, str):
            if aggregator not in _aggregator_registry:
                raise ValueError(f"Unknown aggregator: {aggregator}. Available options: {list(_aggregator_registry.keys())}")
            aggregator_class = _aggregator_registry[aggregator]
            aggregator = aggregator_class()
        
        self.feature_groups = feature_groups
        self.total_features = total_features
        self.group_names = list(feature_groups.keys())
        self.num_groups = len(feature_groups)
        
        # Find ungrouped features (not in any user-defined group)
        all_grouped_indices = set()
        for indices in feature_groups.values():
            all_grouped_indices.update(indices)
        
        self.ungrouped_indices = sorted([i for i in range(total_features) if i not in all_grouped_indices])
        self.num_ungrouped = len(self.ungrouped_indices)
        
        # Global tree input size = number of groups + number of ungrouped features
        self.num_global_inputs = self.num_groups + self.num_ungrouped
        
        # Transformation layer settings
        self.use_sub_tree_transformation = use_sub_tree_transformation
        self.use_global_transformation = use_global_transformation
        self.sub_tree_transformations = sub_tree_transformations
        self.global_tree_transformations = global_tree_transformations
        
        # Learning rates
        self.lr_permutation = lr_permutation
        self.lr_transformation = lr_transformation
        self.lr_aggregator = lr_aggregator
        self.lr_other = lr_other
        self.use_class_weighting = use_class_weighting
        self.loss_amplifier = loss_amplifier
        
        # Temperature annealing parameters (aligned with baconNet)
        self.permutation_initial_temperature = permutation_initial_temperature
        self.permutation_final_temperature = permutation_final_temperature
        self.transformation_initial_temperature = transformation_initial_temperature
        self.transformation_final_temperature = transformation_final_temperature
        
        # Sub-tree permutation settings
        self.sub_tree_permutation_groups = set(sub_tree_permutation_groups) if sub_tree_permutation_groups else set()
        
        # Create sub-trees for ALL user-defined groups
        # Ungrouped features bypass sub-tree overhead and join global tree directly
        self.sub_trees = nn.ModuleDict()
        for group_name, feature_indices in self.feature_groups.items():
            num_features_in_group = len(feature_indices)
            
            # Prepare transformations for this sub-tree
            sub_tree_trans = None
            if use_sub_tree_transformation and sub_tree_transformations is not None:
                # Import transformation classes at runtime
                from bacon.transformationLayer import IdentityTransformation, NegationTransformation
                # Create instances with correct size for this sub-tree
                sub_tree_trans = [trans.__class__(num_features_in_group) for trans in sub_tree_transformations]
            
            # Determine if this group should learn permutations
            enable_permutation = group_name in self.sub_tree_permutation_groups
            
            # Each sub-tree: transformations + aggregation + optional permutation learning
            sub_tree = binaryTreeLogicNet(
                input_size=num_features_in_group,
                freeze_loss_threshold=freeze_loss_threshold if enable_permutation else 999.9,
                weight_mode=weight_mode,
                tree_layout=sub_tree_layout,
                aggregator=aggregator,
                use_transformation_layer=use_sub_tree_transformation,
                transformations=sub_tree_trans,
                is_frozen=not enable_permutation,  # Frozen if NOT learning permutation
                permutation_max=max_permutations if enable_permutation else 1,
                normalize_andness=True,
                weight_penalty_strength=1e-3,
                loss_amplifier=loss_amplifier
            )
            
            # Force identity permutation ONLY if NOT learning permutation
            if not enable_permutation and hasattr(sub_tree, 'input_to_leaf'):
                identity_perm = torch.arange(num_features_in_group)
                sub_tree.input_to_leaf = frozenInputToLeaf(
                    identity_perm, 
                    num_features_in_group
                )
                sub_tree.locked_perm = identity_perm
                sub_tree.is_frozen = True
            
            self.sub_trees[group_name] = sub_tree
        
        # Prepare transformations for global tree
        global_tree_trans = None
        if use_global_transformation and global_tree_transformations is not None:
            # Import transformation classes at runtime
            from bacon.transformationLayer import IdentityTransformation, NegationTransformation
            # Create instances with correct size for global tree
            global_tree_trans = [trans.__class__(self.num_global_inputs) for trans in global_tree_transformations]
        
        # Create global tree that combines group outputs + single features
        # This tree learns: (1) permutation, (2) transformations, (3) aggregation
        self.global_tree = binaryTreeLogicNet(
            input_size=self.num_global_inputs,  # Groups + single features
            freeze_loss_threshold=freeze_loss_threshold,
            weight_mode=weight_mode,
            tree_layout=global_tree_layout,
            aggregator=aggregator,
            use_transformation_layer=use_global_transformation,  # Transform all inputs
            transformations=global_tree_trans,
            is_frozen=False,  # Learn permutation
            permutation_max=max_permutations,
            normalize_andness=True,
            weight_penalty_strength=1e-3,
            loss_amplifier=loss_amplifier
        )
    
    def forward(self, x):
        """
        Forward pass through hierarchical BACON.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, total_features)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        global_inputs = []
        
        # Process ALL user-defined groups through sub-trees
        for group_name in self.group_names:
            feature_indices = self.feature_groups[group_name]
            # Extract features for this group
            group_input = x[:, feature_indices]
            
            # Pass through sub-tree (outputs single value per sample)
            group_output = self.sub_trees[group_name](group_input)
            global_inputs.append(group_output)
        
        # Add ungrouped features directly (no sub-tree overhead)
        for feature_idx in self.ungrouped_indices:
            # Extract single feature and add dimension for consistency
            ungrouped_feature = x[:, feature_idx:feature_idx+1]
            global_inputs.append(ungrouped_feature)
        
        # Stack all inputs for global tree: (batch_size, num_groups + num_ungrouped)
        global_tensor = torch.cat(global_inputs, dim=1)
        
        # Pass through global tree
        final_output = self.global_tree(global_tensor)
        
        return final_output
    
    def train_model(self, x, y, epochs):
        """
        Train the hierarchical BACON network.
        
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            epochs (int): Number of epochs
        
        Returns:
            dict: Training output
        """
        # Training happens through find_best_model
        raise NotImplementedError("Use find_best_model() for training")
    
    def inference(self, x, threshold=0.5):
        """Perform inference with threshold."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > threshold).float()
            return predictions
    
    def inference_raw(self, x):
        """Perform raw inference (returns continuous [0,1] values)."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs
    
    def evaluate(self, x, y, threshold=0.5):
        """Evaluate model accuracy."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > threshold).float()
            accuracy = (predictions == y).float().mean()
            return accuracy.item()
    
    def save_model(self, filepath):
        """Save the hierarchical BACON model."""
        import os
        import logging
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        checkpoint = {
            'feature_groups': self.feature_groups,
            'global_tree_state': self.global_tree.state_dict(),
            'sub_trees_state': {name: tree.state_dict() for name, tree in self.sub_trees.items()},
            'global_is_frozen': self.global_tree.is_frozen,
            'global_locked_perm': self.global_tree.locked_perm,
            'sub_trees_frozen': {name: tree.is_frozen for name, tree in self.sub_trees.items()},
            'sub_trees_locked_perm': {name: tree.locked_perm for name, tree in self.sub_trees.items()},
            # Save architecture configuration
            'use_global_transformation': self.use_global_transformation,
            'use_sub_tree_transformation': self.use_sub_tree_transformation,
            'num_global_transforms': len(self.global_tree.transformation_layer.transformations) if self.global_tree.transformation_layer else 0,
            'num_sub_tree_transforms': {name: len(tree.transformation_layer.transformations) if tree.transformation_layer else 0 for name, tree in self.sub_trees.items()},
        }
        
        logging.info(f"\n💾 SAVING MODEL TO: {filepath}")
        logging.info(f"   Global tree frozen: {self.global_tree.is_frozen}")
        logging.info(f"   Global locked_perm: {self.global_tree.locked_perm}")
        logging.info(f"   use_global_transformation: {self.use_global_transformation}")
        if self.global_tree.transformation_layer:
            trans_summary = self.global_tree.transformation_layer.get_transformation_summary()
            logging.info(f"   Global transformations: {len(self.global_tree.transformation_layer.transformations)} types, {len(trans_summary)} features")
        logging.info(f"   Sub-trees frozen: {checkpoint['sub_trees_frozen']}")
        
        torch.save(checkpoint, filepath)
        logging.info(f"   ✅ Save complete!")
    
    def load_model(self, filepath):
        """Load the hierarchical BACON model."""
        import logging
        checkpoint = torch.load(filepath, weights_only=False)
        
        logging.info(f"\n📂 LOADING MODEL FROM: {filepath}")
        logging.info(f"   Checkpoint global frozen: {checkpoint.get('global_is_frozen', False)}")
        logging.info(f"   Checkpoint global locked_perm: {checkpoint.get('global_locked_perm', None)}")
        logging.info(f"   Checkpoint sub-trees frozen: {checkpoint.get('sub_trees_frozen', {})}")
        
        # Check architecture compatibility
        saved_use_global_trans = checkpoint.get('use_global_transformation', True)
        saved_num_global_trans = checkpoint.get('num_global_transforms', 6)
        current_num_global_trans = len(self.global_tree.transformation_layer.transformations) if self.global_tree.transformation_layer else 0
        
        if saved_use_global_trans != self.use_global_transformation:
            logging.warning(f"   ⚠️  ARCHITECTURE MISMATCH: Checkpoint has use_global_transformation={saved_use_global_trans}, current model has {self.use_global_transformation}")
            logging.warning(f"       This will cause performance degradation!")
        
        if self.use_global_transformation and saved_num_global_trans != current_num_global_trans:
            logging.warning(f"   ⚠️  TRANSFORMATION COUNT MISMATCH: Checkpoint has {saved_num_global_trans} global transforms, current model has {current_num_global_trans}")
            logging.warning(f"       Transformation layer will be randomly reinitialized!")
        
        # Restore global tree
        self.global_tree.is_frozen = checkpoint.get('global_is_frozen', False)
        self.global_tree.locked_perm = checkpoint.get('global_locked_perm', None)
        
        if self.global_tree.is_frozen and self.global_tree.locked_perm is not None:
            from bacon.frozonInputToLeaf import frozenInputToLeaf
            self.global_tree.input_to_leaf = frozenInputToLeaf(
                self.global_tree.locked_perm,
                self.global_tree.original_input_size
            ).to(next(self.parameters()).device)
        
        # Load global tree state, filtering out incompatible transformation parameters
        global_state = checkpoint['global_tree_state']
        current_state = self.global_tree.state_dict()
        
        # Filter out keys that don't match in size
        filtered_state = {}
        skipped_keys = []
        for k, v in global_state.items():
            if k in current_state:
                if current_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {current_state[k].shape})")
            else:
                skipped_keys.append(f"{k} (not in current model)")
        
        if skipped_keys:
            logging.warning(f"   ⚠️  SKIPPED {len(skipped_keys)} global tree parameters:")
            for key in skipped_keys[:5]:  # Show first 5
                logging.warning(f"      - {key}")
            if len(skipped_keys) > 5:
                logging.warning(f"      ... and {len(skipped_keys) - 5} more")
        
        logging.info(f"   ✅ Loading {len(filtered_state)}/{len(global_state)} global tree parameters")
        self.global_tree.load_state_dict(filtered_state, strict=False)
        
        # Restore sub-trees
        sub_trees_frozen = checkpoint.get('sub_trees_frozen', {})
        sub_trees_locked_perm = checkpoint.get('sub_trees_locked_perm', {})
        
        for name, state in checkpoint['sub_trees_state'].items():
            # Restore frozen state for each sub-tree
            if name in sub_trees_frozen:
                self.sub_trees[name].is_frozen = sub_trees_frozen[name]
            if name in sub_trees_locked_perm:
                self.sub_trees[name].locked_perm = sub_trees_locked_perm[name]
                
                if self.sub_trees[name].is_frozen and self.sub_trees[name].locked_perm is not None:
                    from bacon.frozonInputToLeaf import frozenInputToLeaf
                    self.sub_trees[name].input_to_leaf = frozenInputToLeaf(
                        self.sub_trees[name].locked_perm,
                        self.sub_trees[name].original_input_size
                    ).to(next(self.parameters()).device)
            
            # Filter out incompatible transformation parameters
            current_sub_state = self.sub_trees[name].state_dict()
            filtered_sub_state = {}
            sub_skipped_keys = []
            for k, v in state.items():
                if k in current_sub_state:
                    if current_sub_state[k].shape == v.shape:
                        filtered_sub_state[k] = v
                    else:
                        sub_skipped_keys.append(f"{k} (shape mismatch)")
                else:
                    sub_skipped_keys.append(f"{k} (not in current model)")
            
            if sub_skipped_keys:
                logging.warning(f"   ⚠️  Sub-tree '{name}' skipped {len(sub_skipped_keys)} parameters")
            
            logging.info(f"   ✅ Sub-tree '{name}': loaded {len(filtered_sub_state)}/{len(state)} parameters")
            self.sub_trees[name].load_state_dict(filtered_sub_state, strict=False)
    
    def find_best_model(self, x, y, x_test, y_test,
                        attempts=100,
                        acceptance_threshold=0.95,
                        save_path="./hierarchical_bacon.pth",
                        max_epochs=12000,
                        save_model=True,
                        use_hierarchical_permutation=False,
                        hierarchical_group_size=3,
                        hierarchical_epochs_per_attempt=None,
                        hierarchical_bleed_ratio=0.1,
                        hierarchical_bleed_decay=2.0):
        """
        Find best model by training multiple times.
        Similar to baconNet.find_best_model but adapted for hierarchical structure.
        
        Args:
            use_hierarchical_permutation (bool): Use coarse-grained permutation for global tree. Defaults to False.
            hierarchical_group_size (int): Group size for hierarchical permutation. Defaults to 3.
            hierarchical_epochs_per_attempt (int): Epochs per coarse permutation. Defaults to None (uses max_epochs).
            hierarchical_bleed_ratio (float): Bleed ratio between blocks. Defaults to 0.1.
            hierarchical_bleed_decay (float): Bleed decay with distance. Defaults to 2.0.
        """
        import os
        
        best_accuracy = 0.0
        best_model = None
        
        # Check for existing model
        if os.path.exists(save_path):
            try:
                logging.info(f"📂 Found saved model at {save_path}, loading...")
                self.load_model(save_path)
                acc = self.evaluate(x_test, y_test)
                logging.info(f"✅ Loaded model accuracy: {acc:.4f}")
                if acc >= acceptance_threshold:
                    return self.state_dict(), acc
            except Exception as e:
                logging.warning(f"⚠️ Failed to load model: {e}")
        
        device = next(self.parameters()).device
        
        # Determine how many attempts and what permutations to use for global tree
        if use_hierarchical_permutation and hasattr(self.global_tree, 'input_to_leaf'):
            from bacon.inputToLeafSinkhorn import inputToLeafSinkhorn
            n = self.num_global_inputs
            coarse_perms = inputToLeafSinkhorn.generate_all_coarse_permutations(n, hierarchical_group_size)
            total_attempts = len(coarse_perms)
            epochs_per_attempt = hierarchical_epochs_per_attempt if hierarchical_epochs_per_attempt else max_epochs
            logging.info(f"🔀 Hierarchical permutation mode for global tree: {total_attempts} coarse permutations (group_size={hierarchical_group_size})")
            logging.info(f"   Each coarse permutation will train for {epochs_per_attempt} epochs")
        else:
            coarse_perms = [None] * attempts
            total_attempts = attempts
            epochs_per_attempt = max_epochs
        
        # Training loop (simplified, focuses on global tree permutation)
        for attempt in range(total_attempts):
            if use_hierarchical_permutation and coarse_perms[attempt] is not None:
                logging.info(f"🔥 Hierarchical BACON attempt {attempt + 1}/{total_attempts}: coarse perm {coarse_perms[attempt]}")
            else:
                logging.info(f"🔥 Hierarchical BACON attempt {attempt + 1}/{total_attempts}")
            
            # Re-initialize global tree for new attempt
            from bacon.aggregators.lsp import HalfWeightAggregator
            from bacon.aggregators.bool import MinMaxAggregator
            
            aggregator_map = {
                "lsp.half_weight": HalfWeightAggregator(),
                "bool.min_max": MinMaxAggregator()
            }
            
            cfg = self.global_tree
            self.global_tree = binaryTreeLogicNet(
                input_size=self.num_global_inputs,  # Groups + single features
                freeze_loss_threshold=cfg.freeze_loss_threshold / cfg.loss_amplifier,
                weight_mode=cfg.weight_mode,
                tree_layout=cfg.tree_layout,
                aggregator=cfg.aggregator,
                use_transformation_layer=cfg.use_transformation_layer,
                transformation_temperature=cfg.transformation_layer.temperature if cfg.transformation_layer else 1.0,
                transformation_use_gumbel=cfg.transformation_layer.use_gumbel if cfg.transformation_layer else False,
                transformations=cfg._custom_transformations if hasattr(cfg, '_custom_transformations') else None,  # Preserve custom transformations
                is_frozen=False,
                permutation_max=cfg.permutation_max,
                device=device
            )
            
            # Initialize permutation matrix with coarse-grained structure if using hierarchical mode
            if use_hierarchical_permutation and coarse_perms[attempt] is not None:
                if hasattr(self.global_tree, 'input_to_leaf') and hasattr(self.global_tree.input_to_leaf, 'initialize_from_coarse_permutation'):
                    self.global_tree.input_to_leaf.initialize_from_coarse_permutation(
                        coarse_perms[attempt], 
                        group_size=hierarchical_group_size,
                        block_std=0.5,
                        bleed_ratio=hierarchical_bleed_ratio,
                        bleed_decay=hierarchical_bleed_decay
                    )
                    bleed_desc = "hard blocks" if hierarchical_bleed_ratio == 0 else f"bleed={hierarchical_bleed_ratio:.2f}"
                    logging.info(f"   🎯 Initialized global tree permutation with coarse structure ({bleed_desc})")
            
            # Setup optimizer with parameter groups
            param_groups = []
            
            # Global permutation
            if hasattr(self.global_tree, 'input_to_leaf') and hasattr(self.global_tree.input_to_leaf, 'logits'):
                logging.info(f"   ✅ Global tree has Sinkhorn permutation layer")
                logging.info(f"      Logits shape: {self.global_tree.input_to_leaf.logits.shape}")
                # Get hard permutation from soft Sinkhorn matrix using Hungarian algorithm
                from scipy.optimize import linear_sum_assignment
                soft_perm = torch.softmax(self.global_tree.input_to_leaf.logits, dim=1)
                soft_perm_np = soft_perm.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-soft_perm_np)  # maximize
                hard_perm = col_ind[row_ind.argsort()]  # sort by row to get permutation
                logging.info(f"      Initial permutation: {hard_perm}")
                param_groups.append({
                    'params': [self.global_tree.input_to_leaf.logits],
                    'lr': self.lr_permutation,
                    'name': 'global_permutation'
                })
            else:
                logging.warning(f"   ⚠️  Global tree does NOT have learnable permutation!")
                logging.warning(f"      has input_to_leaf: {hasattr(self.global_tree, 'input_to_leaf')}")
                if hasattr(self.global_tree, 'input_to_leaf'):
                    logging.warning(f"      input_to_leaf type: {type(self.global_tree.input_to_leaf)}")
                    logging.warning(f"      has logits: {hasattr(self.global_tree.input_to_leaf, 'logits')}")
            
            # Global transformations
            if self.global_tree.transformation_layer is not None:
                trans_params = list(self.global_tree.transformation_layer.parameters())
                if trans_params:
                    param_groups.append({
                        'params': trans_params,
                        'lr': self.lr_transformation,
                        'name': 'global_transformation'
                    })
            
            # Sub-tree transformations (feature-level)
            for group_name, sub_tree in self.sub_trees.items():
                if sub_tree.transformation_layer is not None:
                    sub_trans_params = list(sub_tree.transformation_layer.parameters())
                    if sub_trans_params:
                        param_groups.append({
                            'params': sub_trans_params,
                            'lr': self.lr_transformation,
                            'name': f'subtree_{group_name}_transformation'
                        })
            
            # Other parameters
            registered_params = set()
            for group in param_groups:
                for p in group['params']:
                    registered_params.add(id(p))
            
            other_params = [p for p in self.parameters() if id(p) not in registered_params]
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'lr': self.lr_other,
                    'name': 'other'
                })
            
            optimizer = torch.optim.Adam(param_groups)
            
            # Log learning rates
            logging.info(f"   📚 Learning rates:")
            for group in param_groups:
                logging.info(f"      {group['name']}: {group['lr']}")
            
            # Setup loss function with optional class weighting
            if self.use_class_weighting:
                pos_count = y.sum().item()
                neg_count = len(y) - pos_count
                if pos_count > 0:
                    pos_weight = neg_count / pos_count
                    logging.info(f"   ⚖️  Class weighting enabled: {pos_count} positives, {neg_count} negatives")
                    logging.info(f"   ⚖️  Positive class weight: {pos_weight:.2f}x")
                    criterion = nn.BCELoss(reduction='none')
                else:
                    criterion = nn.BCELoss()
                    pos_weight = None
            else:
                logging.info(f"   ⚖️  Class weighting disabled")
                criterion = nn.BCELoss()
                pos_weight = None
            
            # Temperature annealing for permutation learning
            perm_initial_temp = self.permutation_initial_temperature
            perm_final_temp = self.permutation_final_temperature
            perm_temp_decay = (perm_final_temp / perm_initial_temp) ** (1.0 / epochs_per_attempt)
            if hasattr(self.global_tree.input_to_leaf, 'temperature'):
                self.global_tree.input_to_leaf.temperature = perm_initial_temp
                logging.info(f"   🌡️  Permutation annealing: {perm_initial_temp:.1f} → {perm_final_temp:.1f} over {epochs_per_attempt} epochs")
            
            # Temperature annealing for transformation selection (converge to single choice)
            # Transformation has 2^n states vs permutation's n! states, so starts cooler
            trans_initial_temp = self.transformation_initial_temperature
            trans_final_temp = self.transformation_final_temperature
            trans_temp_decay = (trans_final_temp / trans_initial_temp) ** (1.0 / epochs_per_attempt)
            
            # Set initial temperature for all transformation layers
            trans_layers = []
            if self.global_tree.transformation_layer is not None:
                self.global_tree.transformation_layer.temperature = trans_initial_temp
                trans_layers.append(('global', self.global_tree.transformation_layer))
            for name, tree in self.sub_trees.items():
                if tree.transformation_layer is not None:
                    tree.transformation_layer.temperature = trans_initial_temp
                    trans_layers.append((name, tree.transformation_layer))
            
            if trans_layers:
                logging.info(f"   🌡️  Transformation annealing: {trans_initial_temp:.1f} → {trans_final_temp:.1f} over {epochs_per_attempt} epochs")
                logging.info(f"      Annealing {len(trans_layers)} transformation layers (forces single transformation per feature)")
            
            # Training loop
            for epoch in range(epochs_per_attempt):
                self.train()
                optimizer.zero_grad(set_to_none=True)
                
                outputs = self.forward(x)
                
                # Compute loss
                if self.use_class_weighting and pos_weight is not None:
                    bce_losses = criterion(outputs, y)
                    weights = torch.where(y == 1, pos_weight, 1.0)
                    loss = (bce_losses * weights).mean() * self.loss_amplifier
                else:
                    loss = criterion(outputs, y) * self.loss_amplifier
                
                loss.backward()
                optimizer.step()
                
                # Anneal permutation temperature (start soft, end hard)
                if hasattr(self.global_tree.input_to_leaf, 'temperature') and not self.global_tree.is_frozen:
                    self.global_tree.input_to_leaf.temperature *= perm_temp_decay
                
                # Anneal transformation temperatures (force convergence to single choice per feature)
                for layer_name, trans_layer in trans_layers:
                    trans_layer.temperature *= trans_temp_decay
                
                # Auto-freeze global tree permutation when loss threshold is reached
                if not self.global_tree.is_frozen and loss.item() < self.global_tree.freeze_loss_threshold:
                    from bacon.frozonInputToLeaf import frozenInputToLeaf
                    from scipy.optimize import linear_sum_assignment
                    # Get current best permutation from Sinkhorn using Hungarian algorithm
                    if hasattr(self.global_tree, 'input_to_leaf') and hasattr(self.global_tree.input_to_leaf, 'logits'):
                        soft_perm = torch.softmax(self.global_tree.input_to_leaf.logits, dim=1)
                        soft_perm_np = soft_perm.detach().cpu().numpy()
                        row_ind, col_ind = linear_sum_assignment(-soft_perm_np)
                        hard_perm = torch.tensor(col_ind[row_ind.argsort()], dtype=torch.long)
                        self.global_tree.locked_perm = hard_perm.clone().detach()
                        self.global_tree.is_frozen = True
                        # Replace Sinkhorn with frozen permutation
                        self.global_tree.input_to_leaf = frozenInputToLeaf(
                            self.global_tree.locked_perm,
                            self.global_tree.original_input_size
                        ).to(self.global_tree.device)
                        logging.info(f"   🔒 Froze global tree permutation at epoch {epoch + 1}, loss: {loss.item():.4f}")
                        logging.info(f"      Locked permutation: {self.global_tree.locked_perm.cpu().numpy()}")
                
                # Logging
                if (epoch + 1) % 500 == 0:
                    logging.info(f"   Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}")
                
                # Early stop if loss is very small
                if loss.item() < 0.1:
                    logging.info(f"   ✅ Early stop: loss {loss.item():.4f} < 0.1")
                    break
            
            # Force freeze permutation at end of training if not already frozen
            if not self.global_tree.is_frozen:
                from bacon.frozonInputToLeaf import frozenInputToLeaf
                from scipy.optimize import linear_sum_assignment
                if hasattr(self.global_tree, 'input_to_leaf') and hasattr(self.global_tree.input_to_leaf, 'logits'):
                    # Get hard permutation from soft Sinkhorn matrix using Hungarian algorithm
                    soft_perm = torch.softmax(self.global_tree.input_to_leaf.logits, dim=1)
                    soft_perm_np = soft_perm.detach().cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(-soft_perm_np)
                    final_perm = torch.tensor(col_ind[row_ind.argsort()], dtype=torch.long)
                    logging.info(f"   📊 Final permutation: {final_perm.cpu().numpy()}")
                    self.global_tree.locked_perm = final_perm.clone().detach()
                    self.global_tree.is_frozen = True
                    # Replace Sinkhorn with frozen permutation
                    self.global_tree.input_to_leaf = frozenInputToLeaf(
                        self.global_tree.locked_perm,
                        self.global_tree.original_input_size
                    ).to(self.global_tree.device)
                    logging.info(f"   🔒 Froze global tree permutation at end of training (final loss: {loss.item():.4f})")
                else:
                    logging.warning(f"   ⚠️  Cannot freeze - no permutation layer found!")
            
            # Evaluate
            accuracy = self.evaluate(x_test, y_test)
            logging.info(f"✅ Attempt {attempt + 1} accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = {
                    'state_dict': self.state_dict(),
                    'feature_groups': self.feature_groups,
                    # Save frozen states at the moment of best accuracy
                    'global_is_frozen': self.global_tree.is_frozen,
                    'global_locked_perm': self.global_tree.locked_perm,
                    'sub_trees_frozen': {name: tree.is_frozen for name, tree in self.sub_trees.items()},
                    'sub_trees_locked_perm': {name: tree.locked_perm for name, tree in self.sub_trees.items()},
                }
                logging.info(f"   🏆 New best model! Accuracy: {best_accuracy:.4f}")
                
                if save_model and save_path:
                    # Save with the best model's frozen states, not current state
                    checkpoint = {
                        'feature_groups': self.feature_groups,
                        'global_tree_state': self.global_tree.state_dict(),
                        'sub_trees_state': {name: tree.state_dict() for name, tree in self.sub_trees.items()},
                        'global_is_frozen': self.global_tree.is_frozen,
                        'global_locked_perm': self.global_tree.locked_perm,
                        'sub_trees_frozen': {name: tree.is_frozen for name, tree in self.sub_trees.items()},
                        'sub_trees_locked_perm': {name: tree.locked_perm for name, tree in self.sub_trees.items()},
                    }
                    torch.save(checkpoint, save_path)
                    logging.info(f"   💾 Saved to {save_path}")
                
                if best_accuracy >= acceptance_threshold:
                    logging.info(f"   ✅ Acceptance threshold reached")
                    break
        
        if best_model is None:
            raise ValueError("No model met the acceptance threshold.")
        
        self.load_state_dict(best_model['state_dict'])
        
        return best_model, best_accuracy
