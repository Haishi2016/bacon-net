import torch.nn as nn
import torch
from bacon.binaryTreeLogicNet import binaryTreeLogicNet
import logging
import os
from bacon.aggregators.lsp import FullWeightAggregator, HalfWeightAggregator
from bacon.aggregators.bool import MinMaxAggregator

_aggregator_registry = {
    "lsp.full_weight": FullWeightAggregator,
    "lsp.half_weight": HalfWeightAggregator,
    "bool.min_max": MinMaxAggregator
}
class baconNet(nn.Module):
    """
    Represents a BACON network for interpretable decision-making using graded logic.

    Args:
        input_size (int): Number of input features. This is likely to be removed in the future.
        freeze_loss_threshold (float, optional): Loss threshold at which to freeze structure learning. Defaults to 0.07. Not if you are using `loss_amplifier`, this will be multiplied by it.
        lock_loss_tolerance (float, optional): Maximum tolerated accuracy loss when locking the structure. Defaults to 0.04. Not if you are using `loss_amplifier`, this will be multiplied by it.
        tree_layout (str, optional): Layout of the tree. Defaults to "left". Other layouts are not supported yet.
        loss_amplifier (float, optional): Amplifier for the loss. Defaults to 1.
        weight_penalty_strength (float, optional): Penalty strength on weights. Defaults to 1e-3. A strong penalty leads to more balaned weights (closer to 0.5).
        normalize_andness (bool, optional): Whether to normalize andness. Defaults to True. This should set to False if the chosen aggregator, such as `bool.min_max`, already normalizes the andness.
        weight_mode (str, optional): Mode for weight configuration. Defaults to "trainable". Use "fixed" for fixed weights (set to 0.5).
        aggregator (str, optional): Aggregator to be used. Defaults to "lsp.full_weight".
        max_permutations (int, optional): Maximum permutations to explore. Defaults to 10000.
        is_frozen (bool, optional): Whether to freeze the structure. Defaults to False.
        early_stop_threshold_large_inputs (float, optional): Early stop threshold for transformation layers with 10+ inputs. Defaults to 0.1. Lower values require more training but achieve higher accuracy.
        reheat_plateau_window (int, optional): Number of epochs to check for plateau detection. Defaults to 200. Smaller = more aggressive reheating.
        reheat_improvement_threshold (float, optional): Minimum loss improvement over plateau_window to avoid reheating. Defaults to 1.0. Smaller = more aggressive.
        reheat_cooldown (int, optional): Minimum epochs between reheats. Defaults to 300. Prevents oscillation.
        reheat_temperature (float, optional): Temperature to use when reheating. Defaults to 10.0. Higher = more exploration.
        permutation_initial_temperature (float, optional): Starting temperature for permutation annealing. Defaults to 5.0. Higher = more initial exploration.
        permutation_final_temperature (float, optional): Final temperature for permutation annealing. Defaults to 0.1. Lower = harder final permutation.
        transformation_initial_temperature (float, optional): Starting temperature for transformation layer. Defaults to 1.0. Should be lower than permutation since transformation is simpler (2^n vs n! states).
        transformation_final_temperature (float, optional): Final temperature for transformation layer. Defaults to 0.1. Same as permutation final temp.
    """
    def __init__(self, input_size, 
                 freeze_loss_threshold=0.07, 
                 lock_loss_tolerance=0.04, 
                 tree_layout="left", 
                 loss_amplifier=1, 
                 weight_penalty_strength=1e-3,
                 weight_mode="trainable",
                 weight_normalization="minmax",
                 aggregator="lsp.full_weight",
                 normalize_andness=True,
                 max_permutations=10000,
                 is_frozen=False,
                 use_transformation_layer=False,
                 transformation_temperature=1.0,
                 transformation_use_gumbel=False,
                 early_stop_threshold_large_inputs=0.1,
                 reheat_plateau_window=200,
                 reheat_improvement_threshold=1.0,
                 reheat_cooldown=300,
                 reheat_temperature=10.0,
                 permutation_initial_temperature=5.0,
                 permutation_final_temperature=0.1,
                 transformation_initial_temperature=1.0,
                 transformation_final_temperature=0.1):
        super(baconNet, self).__init__()        
        if aggregator not in _aggregator_registry:
            raise ValueError(f"Unknown aggregator: {aggregator}. Available options: {list(_aggregator_registry.keys())}")
        aggregator_class = _aggregator_registry[aggregator]
        aggregator = aggregator_class()
        self.is_frozen = is_frozen
        self.early_stop_threshold_large_inputs = early_stop_threshold_large_inputs
        self.reheat_plateau_window = reheat_plateau_window
        self.reheat_improvement_threshold = reheat_improvement_threshold
        self.reheat_cooldown = reheat_cooldown
        self.reheat_temperature = reheat_temperature
        self.permutation_initial_temperature = permutation_initial_temperature
        self.permutation_final_temperature = permutation_final_temperature
        self.transformation_initial_temperature = transformation_initial_temperature
        self.transformation_final_temperature = transformation_final_temperature
        self.assembler = binaryTreeLogicNet(input_size, 
                                            freeze_loss_threshold=freeze_loss_threshold,
                                            weight_mode=weight_mode,
                                            weight_value=0.5,                                             
                                            weight_range=(0.5, 2.0), 
                                            lock_loss_tolerance=lock_loss_tolerance,
                                            normalize_andness=normalize_andness,
                                            tree_layout=tree_layout,
                                            loss_amplifier=loss_amplifier,
                                            is_frozen = is_frozen,
                                            permutation_max=max_permutations,
                                            weight_normalization=weight_normalization,
                                            aggregator=aggregator,
                                            weight_penalty_strength = weight_penalty_strength,
                                            use_transformation_layer=use_transformation_layer,
                                            transformation_temperature=transformation_temperature,
                                            transformation_use_gumbel=transformation_use_gumbel,
                                            weight_choices=None)
    def forward(self, x):
        """ Forward pass through the BACON network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        output = self.assembler(x)
        return output
    def train_model(self, x, y, epochs):
        """ Train the BACON network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            y (torch.Tensor): Target tensor of shape (batch_size, 1).
            epochs (int): Number of training epochs.
        Returns:
            dict: Training output containing loss and accuracy.
        """
        try:
            output = self.assembler.train_model(x,y, epochs, self.is_frozen)
        except RuntimeError as e:
                # We'll raise the error now as there's only one binaryTreeLogicNet
            raise e
        return output
    def inference(self, x, threshold=0.5):
        """ Perform inference on the BACON network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            threshold (float, optional): Threshold for binarizing the output. Defaults to 0.5.
        Returns:
            torch.Tensor: Binarized output tensor of shape (batch_size, 1).
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            # probs = torch.sigmoid(outputs)       # Convert to probabilities
            # predictions = (probs > 0.5).float() 
            predictions = (outputs > threshold).float()
            return predictions
    def inference_raw(self, x):
        """ Perform raw inference on the BACON network. Returns the level of truth in [0,1] instead of binarized output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Raw output tensor of shape (batch_size, 1).
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs
    def evaluate(self, x, y, threshold=0.5):
        """ Evaluate the BACON network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            y (torch.Tensor): Target tensor of shape (batch_size, 1).
            threshold (float, optional): Threshold for binarizing the output. Defaults to 0.5.
        Returns:
            float: Accuracy of the model on the input data.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            # probs = torch.sigmoid(outputs)       # Convert to probabilities
            # predictions = (probs > 0.5).float() 
            predictions = (outputs > threshold).float()  # Binarize the output to match the target labels (0 or 1)
            accuracy = (predictions == y).float().mean()
            return accuracy.item()
    def save_model(self, filepath):
        """ Save the BACON network model to a file.

        Args:
            filepath (str): Path to save the model.
        """        
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.assembler.save_model(filepath)

    def load_model(self, filepath):
        """ Load the BACON network model from a file.

        Args:
            filepath (str): Path to load the model from.
        """
        self.assembler.load_model(filepath)

    def find_best_model(self, x, y, x_test, y_test, 
                        attempts = 100, 
                        acceptance_threshold = 0.95, 
                        save_path = "./assembler.pth", 
                        max_epochs = 12000, 
                        save_model = True,
                        use_hierarchical_permutation = False,
                        hierarchical_group_size = 3,
                        hierarchical_epochs_per_attempt = None,
                        hierarchical_bleed_ratio = 0.1,
                        hierarchical_bleed_decay = 2.0):
        """ Find the best model by training multiple times and evaluating accuracy.

        Args:
            x (torch.Tensor): Input tensor for training.
            y (torch.Tensor): Target tensor for training.
            x_test (torch.Tensor): Input tensor for testing.
            y_test (torch.Tensor): Target tensor for testing.
            attempts (int, optional): Number of attempts to find the best model. Defaults to 100.
            acceptance_threshold (float, optional): Minimum accuracy to accept a model. Defaults to 0.95.
            save_path (str, optional): Path to save the best model. Defaults to "./assembler.pth".
            max_epochs (int, optional): Maximum epochs for training. Defaults to 12000.
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
                    acc = self.evaluate(x_test, y_test)
                    logging.info(f"✅ Loaded model accuracy: {acc:.4f}")
                    if acc >= acceptance_threshold:
                        return self.assembler.state_dict(), acc
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
                # Re-initialize a fresh assembler for this attempt, preserving config
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
                    lock_loss_tolerance=cfg.lock_loss_tolerance / cfg.loss_amplifier,  # unscale
                    freeze_loss_threshold=cfg.freeze_loss_threshold / cfg.loss_amplifier,  # unscale
                    permutation_max=cfg.permutation_max,
                    tree_layout=cfg.tree_layout,
                    weight_penalty_strength=cfg.weight_penalty_strength,
                    aggregator=cfg.aggregator,
                    early_stop_patience=cfg.early_stop_patience,
                    early_stop_min_delta=cfg.early_stop_min_delta,
                    early_stop_threshold=cfg.early_stop_threshold,
                    use_transformation_layer=cfg.use_transformation_layer,
                    transformation_temperature=cfg.transformation_layer.temperature if cfg.transformation_layer else 1.0,
                    transformation_use_gumbel=cfg.transformation_layer.use_gumbel if cfg.transformation_layer else False,
                    device=cfg.device,
                )

                # Initialize permutation matrix with coarse-grained structure if using hierarchical mode
                if use_hierarchical_permutation and coarse_perms[attempt] is not None:
                    if hasattr(self.assembler, 'input_to_leaf') and hasattr(self.assembler.input_to_leaf, 'initialize_from_coarse_permutation'):
                        self.assembler.input_to_leaf.initialize_from_coarse_permutation(
                            coarse_perms[attempt], 
                            group_size=hierarchical_group_size,
                            block_std=0.5,
                            bleed_ratio=hierarchical_bleed_ratio,
                            bleed_decay=hierarchical_bleed_decay
                        )
                        bleed_desc = "hard blocks" if hierarchical_bleed_ratio == 0 else f"bleed={hierarchical_bleed_ratio:.2f}"
                        logging.info(f"   🎯 Initialized permutation matrix with coarse structure ({bleed_desc})")

                # If using transformation layer, disable auto-refine initially
                # We'll enable it after transformations converge
                if self.assembler.transformation_layer is not None:
                    self.assembler.auto_refine = False
                else:
                    self.assembler.auto_refine = True
                self.assembler._auto_refine_every = 10

                optimizer = torch.optim.Adam(self.assembler.parameters(), lr=0.1)
                criterion = nn.BCELoss()

                # Use epochs_per_attempt instead of max_epochs for hierarchical mode
                actual_max_epochs = epochs_per_attempt

                transformation_converged = False
                loss_history = []
                accuracy_history = []  # Track accuracy for smarter plateau detection
                last_reheat_epoch = -1000  # Track when we last reheated
                min_epochs_between_reheats = 500  # Minimum gap between reheating attempts
                
                # Temperature annealing setup
                use_temperature_annealing = (hasattr(self.assembler, 'input_to_leaf') and 
                                            hasattr(self.assembler.input_to_leaf, 'temperature') and
                                            hasattr(self.assembler.input_to_leaf, 'logits'))
                
                if use_temperature_annealing:
                    # Permutation temperature schedule
                    self.assembler.input_to_leaf.temperature = self.permutation_initial_temperature
                    perm_initial_temp = self.permutation_initial_temperature
                    perm_final_temp = self.permutation_final_temperature
                    perm_temp_decay_rate = (perm_final_temp / perm_initial_temp) ** (1.0 / actual_max_epochs)
                    logging.info(f"   🌡️  Permutation annealing: {perm_initial_temp:.1f} → {perm_final_temp:.1f} over {actual_max_epochs} epochs (decay: {perm_temp_decay_rate:.6f})")
                    
                    # Transformation temperature schedule (faster cooling for simpler problem)
                    # Transformation has 2^n states, permutation has n! states
                    # So transformation should converge faster
                    if self.assembler.transformation_layer is not None:
                        trans_initial_temp = self.transformation_initial_temperature
                        trans_final_temp = self.transformation_final_temperature
                        trans_temp_decay_rate = (trans_final_temp / trans_initial_temp) ** (1.0 / actual_max_epochs)
                        self.assembler.transformation_layer.temperature = trans_initial_temp
                        logging.info(f"   🔗 Transformation annealing: {trans_initial_temp:.1f} → {trans_final_temp:.1f} over {actual_max_epochs} epochs (decay: {trans_temp_decay_rate:.6f})")
                    else:
                        trans_temp_decay_rate = None
                    
                    # Disable Hungarian search when using annealing
                    self.assembler.auto_refine = False
                
                for epoch in range(actual_max_epochs):
                    self.assembler.train()
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.assembler(x, targets=y)
                    loss = criterion(outputs, y) * self.assembler.loss_amplifier

                    # Optional weight regularization (only if trainable)
                    if self.assembler.weight_mode == "trainable":
                        depth_weight_penalty = 0.0
                        for w in self.assembler.weights:
                            depth_weight_penalty += self.assembler.weight_penalty_strength * ((torch.sigmoid(w) - 0.5) ** 2).mean()
                        loss = loss + depth_weight_penalty

                    loss.backward()
                    optimizer.step()
                    
                    # Track loss and accuracy for smarter plateau detection
                    loss_history.append(loss.item())
                    
                    # Compute training accuracy every epoch for plateau detection
                    # This is cheap since we already have outputs
                    with torch.no_grad():
                        train_predictions = (outputs > 0.5).float()
                        train_accuracy = (train_predictions == y).float().mean().item()
                        accuracy_history.append(train_accuracy)
                    
                    # Display epoch progress every 100 epochs
                    if (epoch + 1) % 100 == 0:
                        if use_temperature_annealing:
                            perm_temp = self.assembler.input_to_leaf.temperature
                            if self.assembler.transformation_layer is not None:
                                trans_temp = self.assembler.transformation_layer.temperature
                                logging.info(f"   Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}, Perm: {perm_temp:.3f}, Trans: {trans_temp:.3f}")
                            else:
                                logging.info(f"   Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}, Temp: {perm_temp:.3f}")
                        else:
                            logging.info(f"   Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}")
                    
                    # TEMPERATURE ANNEALING: Gradually cool down both layers
                    if use_temperature_annealing and not self.assembler.is_frozen:
                        # Cool permutation layer
                        self.assembler.input_to_leaf.temperature *= perm_temp_decay_rate
                        
                        # Cool transformation layer independently (faster cooling)
                        if self.assembler.transformation_layer is not None and trans_temp_decay_rate is not None:
                            self.assembler.transformation_layer.temperature *= trans_temp_decay_rate
                            # Don't let transformation temp go below permutation temp
                            if self.assembler.transformation_layer.temperature < self.assembler.input_to_leaf.temperature:
                                self.assembler.transformation_layer.temperature = self.assembler.input_to_leaf.temperature
                        
                        # Freeze when permutation temperature is very low (nearly hard)
                        if self.assembler.input_to_leaf.temperature < perm_final_temp * 1.1:
                            self.assembler.is_frozen = True
                            logging.info(f"   ❄️  Permutation hardened at epoch {epoch + 1} (temp: {self.assembler.input_to_leaf.temperature:.3f})")
                    
                    # ADAPTIVE REHEATING: Check periodically if we should escape current basin
                    # Key insight: Don't wait for plateau - proactively check if we're making good enough progress
                    # Check if we have a learnable permutation layer with temperature
                    can_reheat = False  # DISABLED: Testing if natural annealing works better than reheating
                    
                    # Periodic accuracy check: every N epochs, assess if progress is adequate
                    reheat_check_interval = 500  # Check every 500 epochs (less frequent than window size)
                    
                    if can_reheat and (epoch + 1) % reheat_check_interval == 0:
                        if (epoch - last_reheat_epoch) >= min_epochs_between_reheats and not self.assembler.is_frozen:
                            # Check recent progress over a larger window for more stable detection
                            window_size = min(300, len(accuracy_history))  # Use 300 epochs for smoother trend
                            
                            if window_size > 0:
                                old_accuracy = accuracy_history[-window_size]
                                new_accuracy = accuracy_history[-1]
                                accuracy_improvement = new_accuracy - old_accuracy
                                
                                old_loss = loss_history[-window_size]
                                new_loss = loss_history[-1]
                                loss_improvement = old_loss - new_loss
                                
                                # Debug: Log what we're checking
                                if (epoch + 1) % (reheat_check_interval * 5) == 0:  # Every 500 epochs
                                    logging.info(f"   📊 Reheat check @ epoch {epoch+1}: Acc {new_accuracy:.1%} (Δ{accuracy_improvement:+.1%}), Loss {new_loss:.1f} (Δ{loss_improvement:+.1f})")
                                
                                # Proactive reheating strategy:
                                # Reheat when accuracy plateaus (< 1% improvement over window)
                                # BUT NOT if we've already achieved high accuracy (≥ 99%)
                                # If accuracy stops improving, we're likely stuck in a local optimum
                                
                                accuracy_plateau = accuracy_improvement < 0.01  # Less than 1% improvement
                                loss_plateau = loss_improvement < self.reheat_improvement_threshold
                                accuracy_already_good = new_accuracy >= 0.99  # Don't reheat if we're already at target!
                                
                                # Reheat if BOTH loss and accuracy have plateaued AND accuracy isn't already good
                                # (If accuracy is 99%+, we've found a great solution - don't destroy it!)
                                both_stuck = accuracy_plateau and loss_plateau and not accuracy_already_good
                                
                                should_reheat = both_stuck and epoch > 500  # Give it time to settle first
                                
                                # Debug logging to diagnose reheating
                                if (epoch + 1) % (reheat_check_interval * 5) == 0 and both_stuck:
                                    logging.info(f"   🔍 Conditions: both_stuck={both_stuck}, epoch>{500}={epoch > 500}")
                                    logging.info(f"   🔍 Should reheat: {should_reheat}")
                                
                                if should_reheat:
                                    # ADAPTIVE REHEATING: Temperature based on how stuck we are
                                    # Key insight: If we're close to optimal (high accuracy), gentle reheat
                                    #              If we're far from optimal (low accuracy), strong reheat
                                    old_temp = self.assembler.input_to_leaf.temperature
                                    
                                    # Calculate adaptive reheat temperature using continuous formula
                                    # Multiplier formula: 2.0 + 8.0 * (1 - accuracy)^2
                                    # This gives smooth scaling:
                                    #   accuracy=1.00 → 2.0x (very gentle)
                                    #   accuracy=0.95 → 2.2x (gentle)
                                    #   accuracy=0.90 → 2.8x (moderate)
                                    #   accuracy=0.80 → 4.32x (stronger)
                                    #   accuracy=0.70 → 6.72x (strong)
                                    #   accuracy=0.50 → 10.0x (maximum)
                                    reheat_multiplier = 2.0 + 8.0 * ((1.0 - new_accuracy) ** 2)
                                    adaptive_reheat_temp = min(old_temp * reheat_multiplier, self.reheat_temperature)
                                    
                                    self.assembler.input_to_leaf.temperature = adaptive_reheat_temp
                                    
                                    # Synchronize transformation layer temperature when reheating
                                    # Both layers should re-explore together
                                    if self.assembler.transformation_layer is not None:
                                        self.assembler.transformation_layer.temperature = adaptive_reheat_temp
                                    
                                    self.assembler.is_frozen = False
                                    last_reheat_epoch = epoch
                                    
                                    # Log why we're reheating
                                    logging.info(f"   🔥 Accuracy plateau detected! Acc: {new_accuracy:.1%}, improvement: {accuracy_improvement:.1%} over {window_size} epochs")
                                    logging.info(f"   🔥 Adaptive reheating: temp {old_temp:.3f} → {adaptive_reheat_temp:.1f} (multiplier: {reheat_multiplier if reheat_multiplier else 'max'})")
                                    
                                    # Clear some loss/accuracy history to avoid immediate re-trigger
                                    loss_history = loss_history[-100:]
                                    accuracy_history = accuracy_history[-100:]
                    elif len(loss_history) == self.reheat_plateau_window and not use_temperature_annealing:
                        # Debug: log once why reheating is disabled (only if not using annealing)
                        logging.info(f"   ℹ️  Reheating disabled: input_to_leaf type = {type(self.assembler.input_to_leaf).__name__}")
                    
                    # Check if transformation layer has converged
                    # Only check transformation convergence when transformation temperature is low
                    # This ensures transformation converges based on its own schedule, not permutation
                    if not transformation_converged and self.assembler.transformation_layer is not None:
                        # Check if transformation temperature is near its final value
                        transformation_is_cool = False
                        if use_temperature_annealing:
                            trans_temp = self.assembler.transformation_layer.temperature
                            # Consider transformation cool when temp is within 2x of final (e.g., < 0.2 for final 0.1)
                            transformation_is_cool = trans_temp < (self.transformation_final_temperature * 2.0)
                        else:
                            # For Hungarian search, wait for permutation to freeze
                            transformation_is_cool = self.assembler.is_frozen
                        
                        if transformation_is_cool:
                            # Use adaptive convergence: 65% confidence for 70% of features
                            if self.assembler.transformation_layer.has_converged(confidence_threshold=0.65, min_converged_ratio=0.7):
                                transformation_converged = True
                                logging.info(f"   ✅ Transformation layer converged at epoch {epoch + 1} (permutation hardened)")

                    # Early stop when frozen/hardened and loss is small
                    early_stop_threshold = self.assembler.early_stop_threshold
                    if self.assembler.transformation_layer is not None and self.assembler.original_input_size >= 10:
                        early_stop_threshold = self.early_stop_threshold_large_inputs  # Configurable threshold for large inputs with transformations
                    
                    if self.assembler.is_frozen and loss.item() < (early_stop_threshold * self.assembler.loss_amplifier):
                        logging.info(f"   ✅ Early stop: loss {loss.item():.4f} < threshold {early_stop_threshold * self.assembler.loss_amplifier:.4f}")
                        break
                    
                    # Also early stop if we achieve perfect training accuracy (no need to continue)
                    # BUT only if temperature is low enough that the solution is stable
                    if len(accuracy_history) > 0 and accuracy_history[-1] >= 0.999:
                        # Check if permutation is stable (temperature low or frozen)
                        if use_temperature_annealing:
                            current_perm_temp = self.assembler.input_to_leaf.temperature
                            permutation_stable = current_perm_temp < 1.0  # Temperature cool enough for stable solution
                        else:
                            permutation_stable = self.assembler.is_frozen
                        
                        if permutation_stable:
                            logging.info(f"   ✅ Early stop: perfect training accuracy (100.0%) with stable permutation")
                            # Freeze the model to mark it as ready
                            self.assembler.is_frozen = True
                            break
                        # else: accuracy is 100% but temperature still high - might be transient, keep training

                # Force freeze if not already frozen (for pruning and analysis)
                if not self.assembler.is_frozen:
                    if hasattr(self.assembler.input_to_leaf, 'logits'):
                        logging.info(f"   🔒 Force-freezing model (loss threshold not reached)")
                        from bacon.frozonInputToLeaf import frozenInputToLeaf
                        # Get hard assignment from current soft permutation
                        soft_perm = torch.softmax(self.assembler.input_to_leaf.logits, dim=1)
                        hard_perm = torch.argmax(soft_perm, dim=1)
                        self.assembler.locked_perm = hard_perm.clone().detach()
                        self.assembler.is_frozen = True
                        # Replace with frozen layer
                        self.assembler.input_to_leaf = frozenInputToLeaf(
                            self.assembler.locked_perm,
                            self.assembler.original_input_size
                        ).to(self.assembler.device)

                # Always evaluate each attempt, regardless of freeze status
                # This ensures we track the best permutation even if it didn't fully freeze
                accuracy = self.evaluate(x_test, y_test)
                frozen_status = "frozen" if self.assembler.is_frozen else "unfrozen"
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
        """ Prune the features of the BACON network.
        
        Args:
            features (int): Number of features to prune.
        Returns:
            torch.Tensor: Pruned features.
        """
        return self.assembler.prune_features(features=features)    