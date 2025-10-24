import os
import sys
# Ensure local bacon package is used instead of any installed site-package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from bacon.baconNet import baconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
from bacon.utils import generate_paired_boolean_data
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 4

x, y, expr_info = generate_paired_boolean_data(input_size, repeat_factor=100, randomize=True, device=device)
print(f"➗ Paired Expression: {expr_info['expression_text']}")

bacon = baconNet(input_size,
                 aggregator='bool.min_max',
                 weight_mode='fixed',
                 loss_amplifier=1000,
                 normalize_andness=False,
                 tree_layout="paired")

(best_model, best_accuracy) = bacon.find_best_model(x, y, x, y,
                                                   acceptance_threshold=0.95,
                                                   attempts=10,
                                                   max_epochs=min(input_size * 1000, 8000),
                                                   save_model=False)

print(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")
print_tree_structure(bacon.assembler, expr_info['var_names'], classic_boolean=True)
visualize_tree_structure(bacon.assembler, expr_info['var_names'])


