#!/usr/bin/env python3
"""
Math Expressions Sample

This sample demonstrates using BACON with the OperatorSetAggregator to 
approximate mathematical expressions in regression mode.

Supports expressions with:
- Single-letter variables: a, b, c, ...
- Coefficients: 3a, 2.5b, -c
- Operators: +, -, *, /
- Parentheses: (a+b)*c
- Math functions: sqrt(a), sin(b), cos(c), exp(a), log(b), pow(a,2)
"""

import sys
sys.path.insert(0, '../../')

import re
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, List, Dict, Any

from bacon.baconNet import baconNet
from bacon.visualization import print_tree_structure, visualize_alternating_tree, print_alternating_tree_structure
from bacon.tools import reconstruct_expression, print_reconstructed_expression
from samples.common import train_bacon_model

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================================================================
# CONFIGURATION - Edit these values to customize the experiment
# =============================================================================

# Mathematical expression to approximate
# Examples: "a+b+c", "a*b+c", "3*a+2*b-c", "(a+b)*c", "sqrt(a)+b"
EXPRESSION = "3.14*b+10*c-3*a"

# Dataset configuration
NUM_SAMPLES = 2000           # Number of training samples
INPUT_RANGE = (1, 9)        # Range for input values
INPUT_NOISE_PERCENT = 0.0  # Noise level for inputs (0-100)
OUTPUT_NOISE_PERCENT = 0.0 # Noise level for outputs (0-100)

# Training configuration
EPOCHS = 8000               # Number of training epochs
LEARNING_RATE = 0.01         # Learning rate
TREE_LAYOUT = "alternating"        # Tree structure: "left", "balanced", "full", "alternating"

# Operators to use. None = all operators ["add", "sub", "mul", "div"]
# Full operator set - the system should learn to select the right ones
OPERATOR_NAMES = ["add", "mul", "identity"]  # Simpler set for alternating tree

# Temperature settings
EDGE_INITIAL_TEMPERATURE = 10.0  # Higher = more exploration early on
EDGE_FINAL_TEMPERATURE = 0.1    # Lower = sharper final edge selection

# Operator configuration
# Start with moderate tau - outputs are now clamped so soft blend is stable
OPERATOR_INITIAL_TAU = 3.0   # Moderate tau = some differentiation from start
OPERATOR_FINAL_TAU = 0.1      # Very low = sharp selection at end
OPERATOR_USE_GUMBEL = True

FULL_TREE_DEPTH = 2

# Random seed for reproducibility
SEED = 42

# =============================================================================


def parse_expression(expr: str) -> Tuple[List[str], callable]:
    """
    Parse a mathematical expression and extract variable names.
    
    Supports:
    - Single-letter variables: a, b, c, ...
    - Coefficients: 3*a, 2.5*b, -c (use explicit * for multiplication)
    - Operators: +, -, *, /
    - Parentheses: (a+b)*c
    - Math functions: sqrt(a), sin(b), cos(c), exp(a), log(b), pow(a,2), abs(a)
    
    Note: For coefficients, use explicit multiplication: "3*a + 2*b" not "3a + 2b"
    
    Args:
        expr: Mathematical expression string like "3*a + 2*b - c"
        
    Returns:
        Tuple of (list of variable names, evaluation function)
    """
    # Find all single-letter variables (a-z) that are not part of function names
    # Remove function names first to avoid matching their letters
    func_names = ['sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'pow']
    expr_cleaned = expr
    for fn in func_names:
        expr_cleaned = expr_cleaned.replace(fn, '')
    
    variables = sorted(set(re.findall(r'\b([a-z])\b', expr_cleaned)))
    
    if not variables:
        raise ValueError(f"No variables found in expression: {expr}")
    
    # Create evaluation function
    def evaluate(values: Dict[str, float]) -> float:
        """Evaluate the expression with given variable values."""
        import math
        
        # Create namespace with variables and allowed functions
        namespace = {var: values[var] for var in variables}
        namespace['__builtins__'] = {}  # Security: disable builtins
        
        # Allow math functions
        namespace.update({
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'exp': math.exp,
            'log': math.log,
            'abs': abs,
            'pow': pow,
            'pi': math.pi,
            'e': math.e,
        })
        
        return eval(expr, namespace)
    
    return variables, evaluate


def generate_dataset(
    expression: str,
    num_samples: int = 100,
    input_noise_percent: float = 10.0,
    output_noise_percent: float = 10.0,
    input_range: Tuple[float, float] = (1, 99),
    normalize_output: bool = False,
    seed: int = 42,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Generate a dataset based on a mathematical expression.
    
    Args:
        expression: Mathematical expression like "a+b+c" or "a*b-c"
        num_samples: Number of samples to generate
        input_noise_percent: Noise level for input features (0-100)
        output_noise_percent: Noise level for output values (0-100)
        input_range: Range for input values
        normalize_output: If True, normalize outputs to [0,1]. Use False for arithmetic regression.
        seed: Random seed for reproducibility
        device: PyTorch device
        
    Returns:
        Tuple of (X tensor, y tensor, metadata dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Parse expression
    variables, evaluate_fn = parse_expression(expression)
    num_vars = len(variables)
    
    logging.info(f"📐 Expression: {expression}")
    logging.info(f"📊 Variables: {variables}")
    logging.info(f"📊 Generating {num_samples} samples...")
    
    # Generate clean input data
    X_clean = np.random.uniform(input_range[0], input_range[1], size=(num_samples, num_vars))
    
    # Add input noise
    input_noise_scale = input_noise_percent / 100.0
    if input_noise_scale > 0:
        input_noise = np.random.normal(0, input_noise_scale, size=X_clean.shape)
        X_noisy = X_clean + input_noise
        # Clip to actual input range
        X_noisy = np.clip(X_noisy, input_range[0], input_range[1])
    else:
        X_noisy = X_clean
    
    # Compute clean outputs
    y_clean = np.zeros(num_samples)
    for i in range(num_samples):
        values = {var: X_clean[i, j] for j, var in enumerate(variables)}
        try:
            y_clean[i] = evaluate_fn(values)
        except Exception as e:
            logging.warning(f"Error evaluating sample {i}: {e}")
            y_clean[i] = 0.5
    
    # Optionally normalize outputs to [0, 1] range
    y_min, y_max = y_clean.min(), y_clean.max()
    if normalize_output:
        if y_max - y_min > 1e-8:
            y_normalized = (y_clean - y_min) / (y_max - y_min)
        else:
            y_normalized = np.full_like(y_clean, 0.5)
    else:
        # Use raw output values for arithmetic regression
        y_normalized = y_clean
    
    # Add output noise
    output_noise_scale = output_noise_percent / 100.0
    if output_noise_scale > 0:
        # Scale noise relative to output range
        noise_scale = (y_max - y_min) * output_noise_scale if not normalize_output else output_noise_scale
        output_noise = np.random.normal(0, noise_scale, size=y_normalized.shape)
        y_noisy = y_normalized + output_noise
    else:
        y_noisy = y_normalized
    
    # Convert to tensors
    X = torch.tensor(X_noisy, dtype=torch.float32, device=device)
    y = torch.tensor(y_noisy, dtype=torch.float32, device=device).unsqueeze(1)
    
    metadata = {
        'expression': expression,
        'variables': variables,
        'num_samples': num_samples,
        'input_noise_percent': input_noise_percent,
        'output_noise_percent': output_noise_percent,
        'y_original_range': (float(y_min), float(y_max)),
        'normalized': normalize_output,
    }
    
    logging.info(f"   Input shape: {X.shape}")
    logging.info(f"   Input range: [{input_range[0]}, {input_range[1]}]")
    logging.info(f"   Output range: [{y_noisy.min():.4f}, {y_noisy.max():.4f}]")
    if normalize_output:
        logging.info(f"   Output normalized to [0, 1]")
    logging.info(f"   Input noise: {input_noise_percent}%")
    logging.info(f"   Output noise: {output_noise_percent}%")
    
    return X, y, metadata


def create_regression_model(
    num_inputs: int,
    tree_layout: str = "left",
    loss_amplifier: float = 1.0,
    device: torch.device = None
) -> baconNet:
    """
    Create a BACON model configured for regression using OperatorSetAggregator.
    
    Args:
        num_inputs: Number of input features
        tree_layout: Tree structure ("left", "balanced", "full", "alternating")
        loss_amplifier: Loss amplification factor
        device: PyTorch device
        
    Returns:
        Configured baconNet model
    """
    model = baconNet(
        full_tree_depth=FULL_TREE_DEPTH,  # depth=1: direct input→output connection (simplest)
        input_size=num_inputs,
        aggregator='math.operator_set.arith',
        weight_mode='trainable',  # Allow weights to be learned
        weight_normalization='none',  # Disable normalization for arithmetic
        loss_amplifier=loss_amplifier,
        normalize_andness=False,  # Not using andness for arithmetic
        tree_layout=tree_layout,
        use_transformation_layer=False,  # Keep it simple for math
        use_class_weighting=False,  # Regression mode
        permutation_initial_temperature=5.0,
        permutation_final_temperature=0.5,
        # Lower LR for stability with mul operators (prevents gradient explosion)
        lr_aggregator=0.01,  # Much lower than default 0.1
        lr_other=0.01,
        # Regular MSE loss with normalized outputs (data normalized to [0,1])
        regression_loss_type="mse",
        # Operator regularization: balance exploration vs commitment
        # Outputs are now clamped so soft blend is stable
        loss_weight_operator_sparsity=0.1,  # Light pressure to commit
        loss_weight_operator_l2=0.0,  # Disabled - clamping handles stability
        # Full tree settings - triangular tree with balance penalty
        full_tree_temperature=EDGE_INITIAL_TEMPERATURE,
        full_tree_final_temperature=EDGE_FINAL_TEMPERATURE,
        full_tree_shape="triangle",  # Triangular: [3, 2, 1] - forces natural aggregation
        full_tree_max_egress=1,  # Row-softmax: each input routes to exactly 1 node
        loss_weight_full_tree_egress=0.5,  # Encourage peaked distributions
        loss_weight_full_tree_ingress=0.0,  # No hard cap (allow 2+ inputs)
        loss_weight_full_tree_ingress_balance=50.0,  # Strong penalty to prevent all→same dest
        loss_weight_full_tree_scale_reg=0.5,  # R²-modulated: penalizes extreme scales more when R² is high
        full_tree_concentrate_ingress=False,  # Don't use column-softmax
        full_tree_use_sinkhorn=False,  # Don't use Sinkhorn (triangular tree)
        # Alternating tree settings
        alternating_learn_first_routing=False,  # Learn routing in first layer (Input → Agg0)
        alternating_learn_subsequent_routing=False,  # Learn routing in all layers after first
        alternating_max_egress=1,  # Each input routes to exactly 1 node
        alternating_use_straight_through=False,  # HARD routing (no splits). False = soft (like operators)
        loss_weight_alternating_balance=0.1,  # Light starvation protection
        loss_weight_alternating_egress=0.5,  # Encourage peaked routing (only used if soft routing)
        use_permutation_layer=False,  # Let the tree learn routing directly
    )
    
    # Configure operator selection
    if OPERATOR_NAMES is not None:
        model.assembler.aggregator.op_names = OPERATOR_NAMES
        model.assembler.aggregator.num_ops = len(OPERATOR_NAMES)
        
        # Get correct number of aggregator nodes based on tree layout
        if tree_layout == "alternating" and model.assembler.alternating_tree is not None:
            num_agg_nodes = model.assembler.alternating_tree.num_agg_nodes
        elif tree_layout == "full" and model.assembler.fully_connected_tree is not None:
            num_agg_nodes = sum(model.assembler.fully_connected_tree.layer_widths[1:])
        else:
            num_agg_nodes = model.assembler.aggregator.num_layers
        
        # Uniform initialization - no bias toward any operator
        model.assembler.aggregator.op_logits_per_node = nn.ParameterList(
            [nn.Parameter(torch.zeros(len(OPERATOR_NAMES), device=device)) 
             for _ in range(num_agg_nodes)]
        )
        model.assembler.aggregator.num_layers = num_agg_nodes
    model.assembler.aggregator.use_gumbel = OPERATOR_USE_GUMBEL
    
    # Auto-harden when operators become confident - uses hard selection (no gradient leakage)
    # Set moderate threshold: 0.9 = 90% confidence required to lock
    model.assembler.aggregator.auto_harden_threshold = 0.9
    
    return model


def print_operator_selections(model: baconNet, variables: List[str]):
    """Print which operators were selected at each node."""
    aggregator = model.assembler.aggregator
    
    if hasattr(aggregator, 'op_logits_per_node') and aggregator.op_logits_per_node is not None:
        logging.info("\n🔧 Learned Operator Selections:")
        op_names = aggregator.op_names
        
        for i, logits in enumerate(aggregator.op_logits_per_node):
            probs = torch.softmax(logits, dim=0)
            best_op_idx = torch.argmax(probs).item()
            best_op = op_names[best_op_idx]
            confidence = probs[best_op_idx].item()
            
            probs_str = ", ".join([f"{op}={p:.2f}" for op, p in zip(op_names, probs.tolist())])
            logging.info(f"   Node {i + 1}: {best_op} (conf={confidence:.2f}) [{probs_str}]")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🖥️  Using device: {device}")
    
    # Generate dataset using configuration
    X, y, metadata = generate_dataset(
        expression=EXPRESSION,
        num_samples=NUM_SAMPLES,
        input_noise_percent=INPUT_NOISE_PERCENT,
        output_noise_percent=OUTPUT_NOISE_PERCENT,
        input_range=INPUT_RANGE,
        normalize_output=False,  # Keep raw values - model needs to learn actual arithmetic
        seed=SEED,
        device=device
    )
    
    variables = metadata['variables']
    num_vars = len(variables)
    
    # Split into train/test sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model
    logging.info(f"\n🧠 Creating BACON model with {num_vars} inputs...")
    model = create_regression_model(
        num_inputs=num_vars,
        tree_layout=TREE_LAYOUT,
        device=device
    )
    
    # Train model using standard train_bacon_model with regression mode
    best_model, best_r2 = train_bacon_model(
        model=model,
        X_train=X_train,
        Y_train=y_train,
        X_test=X_test,
        Y_test=y_test,
        attempts=1,  # Single attempt
        max_epochs=EPOCHS,
        acceptance_threshold=1.0,  # R² threshold - high for regression
        task_type="regression",
        use_hierarchical_permutation=False,  # Simple permutation for math
        operator_initial_tau=OPERATOR_INITIAL_TAU,
        operator_final_tau=OPERATOR_FINAL_TAU,
        operator_freeze_min_confidence=0.85,  # Require 85% operator commitment before freeze
        operator_freeze_epochs=0,  # Disable two-phase - operators and edges must co-evolve
        frozen_training_epochs=2000,  # Train longer after freezing to refine weights
    )
    
    # Print results
    print_operator_selections(best_model, variables)
    
    # Print reconstructed expression
    print_reconstructed_expression(best_model, variables, precision=4)
    
    # Print tree structure
    logging.info("\n📋 Learned Tree Structure:")
    print_tree_structure(best_model.assembler, variables)
    
    # Summary
    logging.info(f"\n✅ Summary:")
    logging.info(f"   Expression: {EXPRESSION}")
    logging.info(f"   Variables: {variables}")
    logging.info(f"   Samples: {NUM_SAMPLES}")
    logging.info(f"   Noise: input={INPUT_NOISE_PERCENT}%, output={OUTPUT_NOISE_PERCENT}%")
    logging.info(f"   Best R²: {best_r2:.4f}")
    
    # Debug: Check actual vs predicted for a few samples
    logging.info(f"\n🔍 Debug: Sample predictions vs targets:")
    _, eval_fn = parse_expression(EXPRESSION)
    best_model.eval()
    with torch.no_grad():
        # Test with simple known inputs
        test_inputs = torch.tensor([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
        ], device=device)
        
        predictions = best_model(test_inputs)
        for i, (inp, pred) in enumerate(zip(test_inputs, predictions)):
            # Compute actual target using the expression
            values = {var: inp[j].item() for j, var in enumerate(variables)}
            target = eval_fn(values)
            logging.info(f"   Input: {inp.tolist()} -> Pred: {pred.item():.4f}, Target: {target:.1f}")
    
    # Visualize alternating tree structure
    if TREE_LAYOUT == "alternating" and best_model.assembler.alternating_tree is not None:
        
        logging.info("\n🎨 Opening interactive visualization...")
        
        # Get reconstructed expression for display
        try:
            expr_str = reconstruct_expression(best_model, variables)
        except:
            expr_str = "(reconstruction failed)"
        
        visualize_alternating_tree(
            best_model.assembler,
            variable_names=variables,
            title=f"Learned Tree for: {EXPRESSION}",
            expression=expr_str,
            r2=best_r2,
            show=True,
            save_path="alternating_tree_viz.html"
        )
        
        # Also print ASCII structure
        print_alternating_tree_structure(best_model.assembler, variables)
    else:
        logging.info(f"\n⚠️ SKIPPED VISUALIZATION: TREE_LAYOUT={TREE_LAYOUT}, alternating_tree is None={best_model.assembler.alternating_tree is None}")


if __name__ == "__main__":
    main()
