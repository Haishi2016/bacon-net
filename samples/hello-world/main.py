import sys
sys.path.append('../../')
import torch
from bacon.baconNet import baconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
from bacon.utils import generate_classic_boolean_data
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 3

x, y,  expr_info = generate_classic_boolean_data(input_size, repeat_factor=100, device=device)
print(f"➗ Expression: {expr_info['expression_text']}")

bacon = baconNet(input_size, freeze_loss_threshold=0.03, aggregator='lsp.half_weight', weight_mode='fixed')
(best_model, best_accuracy) = bacon.find_best_model(x, y, x, y, acceptance_threshold=0.95, attempts=10, max_epochs=2000, save_model=False)

print(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")
print_tree_structure(bacon.assembler, expr_info['var_names'], classic_boolean=True)
visualize_tree_structure(bacon.assembler, expr_info['var_names'])