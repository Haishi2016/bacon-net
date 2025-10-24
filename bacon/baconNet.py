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
                 is_frozen=False):
        super(baconNet, self).__init__()        
        if aggregator not in _aggregator_registry:
            raise ValueError(f"Unknown aggregator: {aggregator}. Available options: {list(_aggregator_registry.keys())}")
        aggregator_class = _aggregator_registry[aggregator]
        aggregator = aggregator_class()
        self.is_frozen = is_frozen
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
                        save_model = True):
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

        for attempt in range(attempts):
            logging.info(f"🔥 Attempting to find the best model... {attempt + 1}/{attempts}")

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
                    device=cfg.device,
                )

                # Enable light auto-refinement inside forward
                self.assembler.auto_refine = True
                self.assembler._auto_refine_every = 10

                optimizer = torch.optim.Adam(self.assembler.parameters(), lr=0.1)
                criterion = nn.BCELoss()

                for epoch in range(max_epochs):
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

                    # Simple early stop when frozen and loss small
                    if self.assembler.is_frozen and loss.item() < (self.assembler.early_stop_threshold * self.assembler.loss_amplifier):
                        break

                logging.info(f"✅ Permutation is frozen: {self.assembler.is_frozen}")
                if self.assembler.is_frozen:
                    accuracy = self.evaluate(x_test, y_test)
                    logging.info(f"✅ Attempt {attempt + 1} accuracy: {accuracy:.4f}")
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = self.assembler.state_dict()
                        if best_accuracy >= acceptance_threshold:
                            break
            except RuntimeError as e:
                logging.error(f"🔥 Attempt {attempt + 1} failed with error: {e}")
        if best_model is None:
            raise ValueError("No model met the acceptance threshold.")
        self.assembler.load_state_dict(best_model)
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