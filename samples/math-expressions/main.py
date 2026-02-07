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
from bacon.visualization import print_tree_structure
from bacon.tools import reconstruct_expression, print_reconstructed_expression

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================================================================
# CONFIGURATION - Edit these values to customize the experiment
# =============================================================================

# Mathematical expression to approximate
# Examples: "a+b+c", "a*b+c", "3*a+2*b-c", "(a+b)*c", "sqrt(a)+b"
EXPRESSION = "a + 2* b + 10*c"

# Dataset configuration
NUM_SAMPLES = 2000           # Number of training samples
INPUT_RANGE = (1, 9)        # Range for input values
INPUT_NOISE_PERCENT = 0.0  # Noise level for inputs (0-100)
OUTPUT_NOISE_PERCENT = 0.0 # Noise level for outputs (0-100)

# Training configuration
EPOCHS = 8000               # Number of training epochs
LEARNING_RATE = 0.1        # Learning rate
TREE_LAYOUT = "full"        # Tree structure: "left", "balanced", "full"

# Operators to use. None = all operators ["add", "sub", "mul", "div"]
OPERATOR_NAMES = None 

# Edge weight annealing - start high temp for exploration, anneal down for commitment
# Set lower than EPOCHS to complete annealing early and commit to structure
EDGE_INITIAL_TEMPERATURE = 5.0  # High = softer/uniform edge selection
EDGE_FINAL_TEMPERATURE = 0.5    # Low = sharper/peaked edge selection
EDGE_ANNEAL_EPOCHS = 8000       # Number of epochs to anneal edge temperature (None = use all epochs)

# Operator temperature annealing - start high for exploration, anneal down for commitment
OPERATOR_INITIAL_TAU = 5.0    # High = nearly uniform operator weights (~25% each)
OPERATOR_FINAL_TAU = 0.5      # Low = peaked selection (winner takes most)
OPERATOR_ANNEAL_EPOCHS = 8000 # Number of epochs to anneal tau (None = use all epochs)

# Operator auto-hardening threshold (0.5-1.0, or None to disable)
# When the max operator probability exceeds this threshold, use hard selection
OPERATOR_AUTO_HARDEN_THRESHOLD = None

# Use Gumbel noise for operator selection exploration
OPERATOR_USE_GUMBEL = True

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
        tree_layout: Tree structure ("left", "balanced", "full")
        loss_amplifier: Loss amplification factor
        device: PyTorch device
        
    Returns:
        Configured baconNet model
    """
    model = baconNet(
        full_tree_depth=1,  # depth=1: direct input→output connection (simplest)
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
        # Full tree settings
        full_tree_temperature=EDGE_INITIAL_TEMPERATURE,
        full_tree_final_temperature=EDGE_FINAL_TEMPERATURE,
        full_tree_max_egress=None,  # None = sigmoid selection (no sum constraint)
        loss_weight_full_tree_egress=0.5,
    )
    
    # Configure operator selection hardening to eliminate gradient leakage
    # from asymmetric operators (sub, mul)
    if OPERATOR_NAMES is not None:
        model.assembler.aggregator.op_names = OPERATOR_NAMES
        model.assembler.aggregator.num_ops = len(OPERATOR_NAMES)
        # Reinitialize op_logits with new operator count
        model.assembler.aggregator.op_logits_per_node = nn.ParameterList(
            [nn.Parameter(torch.zeros(len(OPERATOR_NAMES), device=device)) 
             for _ in range(model.assembler.aggregator.num_layers)]
        )
    model.assembler.aggregator.use_gumbel = OPERATOR_USE_GUMBEL
    if OPERATOR_AUTO_HARDEN_THRESHOLD is not None:
        model.assembler.aggregator.auto_harden_threshold = OPERATOR_AUTO_HARDEN_THRESHOLD
    
    return model


def train_regression_model(
    model: baconNet,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 3000,
    learning_rate: float = 0.01,
    print_every: int = 500,
    egress_weight: float = 0.5,  # Weight for egress constraint loss (top-K concentration)
    harden_after_training: bool = False  # Disabled for arithmetic - hardening loses scaling info
) -> Tuple[float, List[float]]:
    """
    Train the BACON model in regression mode using MSE loss with regularization.
    
    Args:
        model: BACON model
        X_train: Training inputs
        y_train: Training targets
        epochs: Number of training epochs
        learning_rate: Learning rate
        print_every: Print progress every N epochs
        egress_weight: Weight for egress constraint loss (top-K concentration)
        harden_after_training: Whether to harden the tree after training
        
    Returns:
        Tuple of (final MSE, loss history)
    """
    model.train()
    
    # Use MSE loss for regression
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    is_full_tree = model.assembler.tree_layout == "full"
    
    logging.info(f"\n🏋️ Training for {epochs} epochs...")
    if is_full_tree:
        max_egress = model.assembler.full_tree_max_egress
        logging.info(f"   Egress constraint: max_egress={max_egress}, weight={egress_weight}")
        edge_anneal = EDGE_ANNEAL_EPOCHS if EDGE_ANNEAL_EPOCHS is not None else epochs
        logging.info(f"   Edge temp annealing: {EDGE_INITIAL_TEMPERATURE} → {EDGE_FINAL_TEMPERATURE} over {edge_anneal} epochs")
    aggregator = model.assembler.aggregator
    if hasattr(aggregator, 'tau'):
        op_anneal = OPERATOR_ANNEAL_EPOCHS if OPERATOR_ANNEAL_EPOCHS is not None else epochs
        logging.info(f"   Operator tau annealing: {OPERATOR_INITIAL_TAU} → {OPERATOR_FINAL_TAU} over {op_anneal} epochs")
        aggregator.tau = OPERATOR_INITIAL_TAU
        # Warmup phase: first 30% of training with reduced regularization
    warmup_epochs = int(epochs * 0.3)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Anneal temperature for full tree (sharper edges as training progresses)
        # Use separate schedule for edge annealing
        edge_anneal_epochs = EDGE_ANNEAL_EPOCHS if EDGE_ANNEAL_EPOCHS is not None else epochs
        edge_progress = min(1.0, epoch / max(1, edge_anneal_epochs - 1))
        if is_full_tree:
            model.assembler.anneal_full_tree_temperature(edge_progress)
            model.assembler.anneal_full_tree_gumbel(edge_progress)
        
        # Anneal operator temperature (high early = exploration, low later = commitment)
        if hasattr(aggregator, 'tau'):
            anneal_epochs = OPERATOR_ANNEAL_EPOCHS if OPERATOR_ANNEAL_EPOCHS is not None else epochs
            op_progress = min(1.0, epoch / max(1, anneal_epochs - 1))
            aggregator.tau = OPERATOR_INITIAL_TAU + op_progress * (OPERATOR_FINAL_TAU - OPERATOR_INITIAL_TAU)
        
        outputs = model(X_train)
        mse_loss = criterion(outputs, y_train)
        
        # Total loss with regularization
        loss = mse_loss * model.assembler.loss_amplifier
        
        # Add full tree regularization (ramped up after warmup)
        if is_full_tree:
            # Ramp factor: 0 during warmup, then linearly increases to 1
            if epoch < warmup_epochs:
                ramp_factor = 0.0
            else:
                ramp_factor = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
            
            # Egress loss: encourage each source to concentrate to top-K destinations
            if egress_weight > 0:
                egress_loss = model.assembler.get_full_tree_egress_loss()
                loss = loss + ramp_factor * egress_weight * egress_loss * model.assembler.loss_amplifier
        
        loss.backward()
        optimizer.step()
        
        loss_value = mse_loss.item()
        loss_history.append(loss_value)
        
        if (epoch + 1) % print_every == 0 or epoch == 0:
            # Compute R² score
            with torch.no_grad():
                predictions = model(X_train)
                ss_res = ((y_train - predictions) ** 2).sum()
                ss_tot = ((y_train - y_train.mean()) ** 2).sum()
                r2 = 1 - (ss_res / ss_tot)
            
            extra_info = ""
            if is_full_tree:
                with torch.no_grad():
                    if egress_weight > 0:
                        eg = model.assembler.get_full_tree_egress_loss().item()
                        extra_info += f", egress={eg:.3f}"
                
            logging.info(f"   Epoch {epoch + 1:5d}: MSE = {loss_value:.6f}, R² = {r2.item():.4f}{extra_info}")
    
    # Harden the tree to discrete selections
    if is_full_tree and harden_after_training:
        logging.info("\n🔨 Hardening tree to discrete edges...")
        model.assembler.harden_full_tree(mode="smart")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_train)
        final_mse = criterion(predictions, y_train).item()
        ss_res = ((y_train - predictions) ** 2).sum()
        ss_tot = ((y_train - y_train.mean()) ** 2).sum()
        final_r2 = 1 - (ss_res / ss_tot)
    
    logging.info(f"\n📊 Final Results:")
    logging.info(f"   MSE: {final_mse:.6f}")
    logging.info(f"   R²:  {final_r2.item():.4f}")
    
    return final_mse, loss_history


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
        normalize_output=False,  # Use raw values for arithmetic regression
        seed=SEED,
        device=device
    )
    
    variables = metadata['variables']
    num_vars = len(variables)
    
    # Create model
    logging.info(f"\n🧠 Creating BACON model with {num_vars} inputs...")
    model = create_regression_model(
        num_inputs=num_vars,
        tree_layout=TREE_LAYOUT,
        device=device
    )
    
    # Train model
    final_mse, loss_history = train_regression_model(
        model=model,
        X_train=X,
        y_train=y,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )
    
    # Print results
    print_operator_selections(model, variables)
    
    # Print reconstructed expression
    print_reconstructed_expression(model, variables)
    
    # Print tree structure
    logging.info("\n📋 Learned Tree Structure:")
    print_tree_structure(model.assembler, variables)
    
    # Summary
    logging.info(f"\n✅ Summary:")
    logging.info(f"   Expression: {EXPRESSION}")
    logging.info(f"   Variables: {variables}")
    logging.info(f"   Samples: {NUM_SAMPLES}")
    logging.info(f"   Noise: input={INPUT_NOISE_PERCENT}%, output={OUTPUT_NOISE_PERCENT}%")
    logging.info(f"   Final MSE: {final_mse:.6f}")
    
    # Debug: Check actual vs predicted for a few samples
    logging.info(f"\n🔍 Debug: Sample predictions vs targets:")
    _, eval_fn = parse_expression(EXPRESSION)
    model.eval()
    with torch.no_grad():
        # Test with simple known inputs
        test_inputs = torch.tensor([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
        ], device=device)
        
        predictions = model(test_inputs)
        for i, (inp, pred) in enumerate(zip(test_inputs, predictions)):
            # Compute actual target using the expression
            values = {var: inp[j].item() for j, var in enumerate(variables)}
            target = eval_fn(values)
            logging.info(f"   Input: {inp.tolist()} -> Pred: {pred.item():.4f}, Target: {target:.1f}")
        
        # Check what input_to_leaf does
        logging.info(f"\n🔍 Debug: input_to_leaf permutation check:")
        leaf_values = model.assembler.input_to_leaf(test_inputs)
        logging.info(f"   Input shape: {test_inputs.shape}, Leaf shape: {leaf_values.shape}")
        logging.info(f"   First input: {test_inputs[0].tolist()}")
        logging.info(f"   After input_to_leaf: {leaf_values[0].tolist()}")
        
        # Show the actual soft permutation matrix
        if hasattr(model.assembler.input_to_leaf, 'weights'):
            with torch.no_grad():
                perm_weights = model.assembler.input_to_leaf.weights
                perm_matrix = torch.softmax(perm_weights, dim=1)
                logging.info(f"\n🔍 Debug: Soft permutation matrix (input → leaf):")
                logging.info(f"   Shape: {perm_matrix.shape}")
                for i in range(perm_matrix.shape[0]):
                    row = perm_matrix[i].tolist()
                    max_idx = perm_matrix[i].argmax().item()
                    logging.info(f"   Leaf {i} ← Input {max_idx} (probs: {[f'{p:.3f}' for p in row]})")
        
        # Check with distinct inputs to see permutation effect
        logging.info(f"\n🔍 Debug: Permutation with distinct inputs [1, 2, 3]:")
        distinct_input = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        distinct_leaf = model.assembler.input_to_leaf(distinct_input)
        logging.info(f"   Input [a=1, b=2, c=3] → Leaves {distinct_leaf[0].tolist()}")


if __name__ == "__main__":
    main()
