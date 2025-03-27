import torch.nn as nn
import torch
from bacon.binaryTreeLogicNet import binaryTreeLogicNet
import logging
class baconNet(nn.Module):
    def __init__(self, input_size=10):
        super(baconNet, self).__init__()
        self.assembler = binaryTreeLogicNet(input_size, weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None)
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
    def evaluate(self, x, y):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > 0.5).float()  # Binarize the output to match the target labels (0 or 1)
            accuracy = (predictions == y).float().mean()
            return accuracy.item()
    def find_best_model(self, x, y, attempts = 100, acceptance_threshold = 0.95):
        best_accuracy = 0.0
        best_model = None
        for attempt in range(attempts):
            logging.info(f"🔥 Attempting to find the best model... {attempt + 1}/{attempts}")
            try:
                self.train_model(x, y, epochs=12000)
                accuracy = self.evaluate(x, y)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = self.assembler.state_dict()
                    if best_accuracy >= acceptance_threshold:
                        break
            except RuntimeError as e:
                logging.error(f"🔥 Attempt {attempt + 1} failed with error: {e}")
        if best_model is None:
            raise ValueError("No model met the acceptance threshold.")
        
        return best_model, best_accuracy