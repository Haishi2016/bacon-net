"""
Test script for Alternating Coefficient-Aggregation Tree

Tests the new architecture that separates coefficient learning from routing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from bacon.alternatingTree import AlternatingTree
from bacon.visualization import visualize_alternating_tree, print_alternating_tree_structure
from bacon.aggregators.math.operator_set import ArithmeticOperatorSet


def generate_data(expression: str, num_samples: int = 2000, input_range=(1, 9), seed=42):
    """Generate training data for an expression."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Parse variable names from expression
    import re
    variables = sorted(set(re.findall(r'\b([a-z])\b', expression)))
    num_vars = len(variables)
    
    # Generate random inputs
    X = np.random.uniform(input_range[0], input_range[1], size=(num_samples, num_vars))
    
    # Evaluate expression
    def evaluate(x_row):
        local_vars = {var: x_row[i] for i, var in enumerate(variables)}
        return eval(expression, {"__builtins__": {}}, local_vars)
    
    y = np.array([evaluate(X[i]) for i in range(num_samples)])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
    
    print(f"📐 Expression: {expression}")
    print(f"📊 Variables: {variables}")
    print(f"📊 Generated {num_samples} samples")
    print(f"   Input range: {input_range}")
    print(f"   Output range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X_tensor, y_tensor, variables


def train_alternating_tree(
    X: torch.Tensor,
    y: torch.Tensor,
    variables: list,
    epochs: int = 5000,
    lr: float = 0.01,
    balance_weight: float = 50.0,
    egress_weight: float = 0.5,
    learn_first_routing: bool = True
):
    """Train the alternating tree model."""
    device = X.device
    num_inputs = X.shape[1]
    
    # Create model
    model = AlternatingTree(
        num_inputs=num_inputs,
        learn_first_routing=learn_first_routing,
        temperature=3.0,
        final_temperature=0.1,
        use_gumbel=True,
        gumbel_noise_scale=1.0,
        device=device
    )
    
    # Create aggregator - use ONLY safe operators initially
    aggregator = ArithmeticOperatorSet(
        op_names=["add", "mul", "identity"],  # No div/sub to avoid instability
        use_gumbel=True,
        tau=3.0,
        output_clamp=1e4
    )
    aggregator.attach_to_tree(model.num_agg_nodes)
    
    # Move aggregator parameters to device
    if aggregator.op_logits_per_node is not None:
        for logits in aggregator.op_logits_per_node:
            logits.data = logits.data.to(device)
    
    # Optimizer - include both model and aggregator parameters
    all_params = list(model.parameters())
    if aggregator.op_logits_per_node is not None:
        all_params.extend(aggregator.op_logits_per_node)
    
    optimizer = optim.Adam(all_params, lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_state = None
    best_epoch = 0
    
    print(f"\n🔥 Training for {epochs} epochs...")
    print(f"   Balance weight: {balance_weight}, Egress weight: {egress_weight}")
    
    for epoch in range(epochs):
        model.train()
        
        # Debug first few epochs
        if epoch < 3:
            print(f"   [Debug] Epoch {epoch}:")
            with torch.no_grad():
                for i, cl in enumerate(model.coeff_layers):
                    coeffs = cl.get_coefficients()
                    print(f"     Coeff layer {i}: {coeffs.cpu().numpy()}")
        
        # Anneal temperature
        progress = epoch / epochs
        model.anneal_temperature(progress)
        model.anneal_gumbel(progress, initial=1.0, final=0.1)
        
        # Anneal operator tau
        new_tau = 3.0 - progress * (3.0 - 0.1)
        aggregator.tau = new_tau
        
        if hasattr(aggregator, 'start_forward'):
            aggregator.start_forward()
        
        optimizer.zero_grad()
        
        outputs = model(X, aggregator=aggregator)
        mse_loss = criterion(outputs, y)
        
        # Structural regularization
        balance_loss = model.get_balance_loss()
        egress_loss = model.get_egress_loss()
        
        # Debug balance loss
        if epoch < 5:
            print(f"     balance_loss: {balance_loss.item()}, egress_loss: {egress_loss.item()}")
        
        loss = mse_loss + balance_weight * balance_loss + egress_weight * egress_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)  # Stronger clipping
        
        # Check for NaN gradients
        has_nan_grad = False
        for p in all_params:
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                p.grad.zero_()
        
        optimizer.step()
        
        if mse_loss.item() < best_loss:
            best_loss = mse_loss.item()
            best_state = {
                'model': {k: v.clone() for k, v in model.state_dict().items()},
                'aggregator': [logits.clone() for logits in aggregator.op_logits_per_node] if aggregator.op_logits_per_node else None
            }
            best_epoch = epoch
        
        if (epoch + 1) % 500 == 0:
            # Compute R²
            with torch.no_grad():
                ss_res = ((y - outputs) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum()
                r2 = (1 - ss_res / ss_tot).item()
            
            op_conf = 0.0
            if aggregator.op_logits_per_node:
                with torch.no_grad():
                    probs = [torch.softmax(logits, dim=0).max().item() for logits in aggregator.op_logits_per_node]
                    op_conf = sum(probs) / len(probs)
            
            current_temp = model.agg_layers[0].temperature if hasattr(model.agg_layers[0], 'temperature') else 0.0
            print(f"   Epoch {epoch+1:5d}/{epochs}, Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, "
                  f"R²: {r2:.4f}, OpConf: {op_conf:.2f}, Temp: {current_temp:.2f}")
    
    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state['model'])
        if best_state['aggregator'] is not None:
            for i, logits in enumerate(aggregator.op_logits_per_node):
                logits.data = best_state['aggregator'][i]
    
    print(f"\n✅ Best MSE: {best_loss:.4f} at epoch {best_epoch}")
    
    # Harden and evaluate
    model.harden()
    
    with torch.no_grad():
        aggregator.start_forward()
        outputs = model(X, aggregator=aggregator)
        ss_res = ((y - outputs) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()
    
    print(f"🏆 Final R²: {r2:.4f}")
    
    # Print structure
    print(model.get_structure_description())
    
    # Print operator selections
    print("\n🔧 Operator Selections:")
    if aggregator.op_logits_per_node:
        for i, logits in enumerate(aggregator.op_logits_per_node):
            probs = torch.softmax(logits, dim=0).detach().cpu()
            op_idx = probs.argmax().item()
            op_name = aggregator.op_names[op_idx]
            conf = probs[op_idx].item()
            prob_str = ", ".join([f"{aggregator.op_names[j]}={probs[j].item():.2f}" for j in range(len(aggregator.op_names))])
            print(f"   Node {i}: {op_name} (conf={conf:.2f}) [{prob_str}]")
    
    # Sample predictions
    print("\n🔍 Sample predictions:")
    test_inputs = [
        [1.0] * num_inputs,
        [2.0] * num_inputs,
        [5.0] * num_inputs,
    ]
    for inp in test_inputs:
        inp_tensor = torch.tensor([inp], dtype=torch.float32, device=device)
        with torch.no_grad():
            aggregator.start_forward()
            pred = model(inp_tensor, aggregator=aggregator)
        # Compute target
        local_vars = {var: inp[i] for i, var in enumerate(variables)}
        target = eval(expression, {"__builtins__": {}}, local_vars)
        print(f"   Input: {inp} -> Pred: {pred.item():.4f}, Target: {target:.4f}")
    
    # Reconstruct expression from tree structure
    learned_expr = reconstruct_expression(model, aggregator, variables)
    print(f"\n📝 Reconstructed expression: {learned_expr}")
    
    return model, aggregator, best_loss, r2, learned_expr


def reconstruct_expression(model, aggregator, variable_names):
    """Reconstruct the expression from the learned tree structure."""
    # Get edge weights from first aggregation layer
    first_agg = model.agg_layers[0]
    if hasattr(first_agg, 'get_edge_weights'):
        edges = first_agg.get_edge_weights().detach().cpu().numpy()
    else:
        edges = first_agg.edges.cpu().numpy()
    
    # Get coefficients
    coeffs_0 = model.coeff_layers[0].get_coefficients().cpu().numpy()
    coeffs_1 = model.coeff_layers[1].get_coefficients().cpu().numpy() if len(model.coeff_layers) > 1 else [1.0, 1.0]
    
    # Get operators
    op_names = []
    if aggregator.op_logits_per_node:
        for logits in aggregator.op_logits_per_node:
            probs = torch.softmax(logits, dim=0).detach().cpu()
            op_idx = probs.argmax().item()
            op_names.append(aggregator.op_names[op_idx])
    
    # Build expression based on routing
    # First agg layer: which inputs go to which node
    node_inputs = {j: [] for j in range(first_agg.out_width)}
    for i in range(first_agg.in_width):
        dest = edges[i].argmax()
        coeff = coeffs_0[i]
        var = variable_names[i]
        if abs(coeff - 1.0) < 0.01:
            node_inputs[dest].append(var)
        else:
            node_inputs[dest].append(f"{coeff:.2f}*{var}")
    
    # Build sub-expressions for each first-layer node
    sub_exprs = []
    for j in range(first_agg.out_width):
        inputs = node_inputs[j]
        op = op_names[j] if j < len(op_names) else "?"
        if len(inputs) == 0:
            sub_exprs.append("0")
        elif len(inputs) == 1:
            sub_exprs.append(inputs[0])
        else:
            if op == "add":
                sub_exprs.append(f"({' + '.join(inputs)})")
            elif op == "mul":
                sub_exprs.append(f"({' * '.join(inputs)})")
            else:
                sub_exprs.append(f"{op}({', '.join(inputs)})")
    
    # Apply second layer coefficients
    final_inputs = []
    for j, expr in enumerate(sub_exprs):
        if j < len(coeffs_1):
            coeff = coeffs_1[j]
            if abs(coeff - 1.0) < 0.01:
                final_inputs.append(expr)
            else:
                final_inputs.append(f"{coeff:.2f}*{expr}")
        else:
            final_inputs.append(expr)
    
    # Final operator (last agg node)
    final_op = op_names[-1] if op_names else "?"
    if final_op == "add":
        result = " + ".join(final_inputs)
    elif final_op == "mul":
        result = " * ".join(final_inputs)
    else:
        result = f"{final_op}({', '.join(final_inputs)})"
    
    return result


if __name__ == "__main__":
    # Test expression: b*c + a
    expression = "b*c+a"
    
    X, y, variables = generate_data(expression, num_samples=2000, input_range=(1, 9))
    
    # Test with learned first routing
    print("\n" + "="*60)
    print("MODE: Learned first routing")
    print("="*60)
    model, aggregator, best_mse, final_r2, learned_expr = train_alternating_tree(
        X, y, variables,
        epochs=8000,
        lr=0.01,
        balance_weight=50.0,
        egress_weight=0.5,
        learn_first_routing=True
    )
    
    # Print ASCII tree structure
    print_alternating_tree_structure(model, aggregator, variables)
    
    # Interactive visualization (opens in browser)
    print("\n🎨 Opening interactive visualization...")
    visualize_alternating_tree(
        model,
        aggregator=aggregator,
        variable_names=variables,
        title=f"Learned Tree for: {expression}",
        expression=learned_expr,
        mse=best_mse,
        r2=final_r2,
        show=True,
        save_path="alternating_tree_viz.html"
    )
