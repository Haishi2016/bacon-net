import torch.nn as nn
import torch
from bacon.binaryTreeLogicNet import binaryTreeLogicNet
import logging
import os

class baconNet(nn.Module):
    def __init__(self, input_size, freeze_loss_threshold=0.07, lock_loss_tolerance=0.01, tree_layout="left"):
        super(baconNet, self).__init__()
        self.assembler = binaryTreeLogicNet(input_size, 
                                            freeze_loss_threshold=freeze_loss_threshold,
                                            weight_mode="trainable", 
                                            weight_value=1.0,                                             
                                            weight_range=(0.5, 2.0), 
                                            lock_loss_tolerance=lock_loss_tolerance,
                                            tree_layout=tree_layout,
                                            weight_choices=None)
    def forward(self, x):
        output = self.assembler(x)
        return output
    def train_model(self, x, y, epochs):
        try:
            output = self.assembler.train_model(x,y, epochs)
        except RuntimeError as e:
                # We'll raise the error now as there's only one binaryTreeLogicNet
            raise e
        return output
    def inference(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > 0.5).float()
            return predictions
    def evaluate(self, x, y):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > 0.5).float()  # Binarize the output to match the target labels (0 or 1)
            accuracy = (predictions == y).float().mean()
            return accuracy.item()
    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"assembler.pth")
        self.assembler.save_model(path)

    def load_model(self, directory):
        path = os.path.join(directory, f"assembler.pth")
        self.assembler.load_model(path)

    def find_best_model(self, x, y, x_test, y_test, attempts = 100, acceptance_threshold = 0.95, save_path = ".", max_epochs = 12000, save_model = True):
        best_accuracy = 0.0
        best_model = None        
        assembler_path = os.path.join(save_path, "assembler.pth")
        if os.path.exists(assembler_path):
            try:
                logging.info(f"📂 Found saved model at {assembler_path}, loading...")
                self.load_model(save_path)
                if self.assembler.is_frozen:
                    acc = self.evaluate(x_test, y_test)
                    logging.info(f"✅ Loaded model accuracy: {acc:.4f}")
                    if acc >= acceptance_threshold:
                        return self.assembler.state_dict(), acc
            except Exception as e:
                logging.warning(f"⚠️ Failed to load model from {assembler_path}: {e}")

        for attempt in range(attempts):
            logging.info(f"🔥 Attempting to find the best model... {attempt + 1}/{attempts}")

            torch.manual_seed(torch.initial_seed() + attempt)

            try:
                self.train_model(x, y, epochs=max_epochs)
                if self.assembler.is_frozen:                   
                    accuracy = self.evaluate(x_test, y_test)
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
            self.save_model(".")
        return best_model, best_accuracy
    
    def print_tree_structure(self, labels=None):
        self.assembler.print_tree_structure(labels)
        print(f"Permutation: {self.assembler.locked_perm}")
    
    def visualize_tree_structure(self, labels=None):
        self.assembler.visualize_tree_structure(labels)
        print(f"Permutation: {self.assembler.locked_perm}")
    def prune_features(self, features):
        return self.assembler.prune_features(features=features)    