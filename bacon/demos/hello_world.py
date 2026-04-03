"""Hello-world demo — discover a classic Boolean expression from synthetic data.

This is a self-contained demo that requires only the bacon package and its
standard dependencies (torch).  No external datasets are downloaded.
"""

import random
import torch
from bacon.baconNet import baconNet
from bacon.visualization import print_tree_structure
from bacon.utils import generate_classic_boolean_data


def run(args):
    """Run the hello-world demo.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments.  Expected attributes:
        - variables (int): number of Boolean input variables
        - seed (int): random seed
    """
    variables = args.variables
    seed = args.seed

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For small variable counts, use the full truth table so the
    # expression is structurally identifiable without sampling noise.
    use_full = variables <= 10
    repeat_factor = 1 if use_full else 100
    randomize = not use_full

    x, y, expr_info = generate_classic_boolean_data(
        variables,
        repeat_factor=repeat_factor,
        randomize=randomize,
        device=device,
    )
    print(f"Expression: {expr_info['expression_text']}")

    model = baconNet(
        variables,
        aggregator='bool.min_max',
        tree_layout='left',
        weight_mode='fixed',
        loss_amplifier=1000,
        normalize_andness=False,
        use_permutation_layer=False,
    )

    best_model, best_accuracy = model.find_best_model(
        x, y, x, y,
        acceptance_threshold=0.95,
        attempts=5,
        max_epochs=min(variables * 300, 8000),
        save_model=False,
    )

    print(f"Best accuracy: {best_accuracy * 100:.2f}%")

    pred = model.inference(x, threshold=0.5)
    final_accuracy = (pred == y).float().mean().item()
    print(f"Final accuracy: {final_accuracy * 100:.2f}%")

    print_tree_structure(
        model.assembler,
        expr_info['var_names'],
        classic_boolean=True,
        layout='left',
    )

    return 0
