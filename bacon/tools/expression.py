"""
Expression Reconstruction Tool

Utilities for reconstructing human-readable mathematical expressions from 
trained BACON models with OperatorSetAggregator.
"""

import torch
import numpy as np
import logging
import re
from typing import List, TYPE_CHECKING

try:
    import sympy
except ImportError:
    sympy = None

if TYPE_CHECKING:
    from bacon.baconNet import baconNet


_SYMBOL_PATTERN = re.compile(r"\b[a-zA-Z]\w*\b")


def reconstruct_expression(
    model: "baconNet", 
    variables: List[str],
    weight_threshold: float = 0.1,
    show_weights: bool = True,
    precision: int = 2
) -> str:
    """
    Reconstruct a human-readable mathematical expression from the trained model.
    
    This function walks the tree structure, extracts the learned operator at each
    node, and combines them with the variable names weighted by the learned weights.
    
    Args:
        model: Trained BACON model with OperatorSetAggregator
        variables: List of variable names (e.g., ['a', 'b', 'c'])
        weight_threshold: Minimum weight to include a term (prune near-zero weights)
        show_weights: If True, show coefficient weights in the expression
        precision: Number of decimal places for weights
        
    Returns:
        String representation of the learned expression
    """
    assembler = model.assembler
    aggregator = assembler.aggregator
    
    # Check if we have an operator set aggregator
    if not hasattr(aggregator, 'op_logits_per_node') or aggregator.op_logits_per_node is None:
        return "Cannot reconstruct: aggregator does not have operator logits"

    if hasattr(assembler, 'get_input_labels'):
        variables = assembler.get_input_labels(variables)
    
    op_names = aggregator.op_names
    num_nodes = len(aggregator.op_logits_per_node)
    
    # Get operator at each node (argmax of softmax)
    operators = []
    for logits in aggregator.op_logits_per_node:
        probs = torch.softmax(logits, dim=0)
        best_idx = torch.argmax(probs).item()
        operators.append(op_names[best_idx])
    
    # Handle permutation layer - reorder variables according to learned permutation
    # This matches how visualization.py handles locked_perm
    if hasattr(assembler, 'locked_perm') and assembler.locked_perm is not None:
        perm = assembler.locked_perm.tolist()
        permuted_variables = [variables[i] for i in perm]
        if len(variables) > len(perm):
            permuted_variables.extend(variables[len(perm):])
    elif hasattr(assembler, 'input_to_leaf') and hasattr(assembler.input_to_leaf, 'weights'):
        # Derive permutation from soft input_to_leaf weights
        # perm_matrix[leaf, input]: probability that leaf gets input
        with torch.no_grad():
            perm_weights = assembler.input_to_leaf.weights
            perm_matrix = torch.softmax(perm_weights, dim=1)
            # For each leaf, find which input it most likely receives
            perm = perm_matrix.argmax(dim=1).tolist()
            permuted_variables = [variables[i] if i < len(variables) else f"x{i}" for i in perm]
            if len(variables) > len(perm):
                permuted_variables.extend(variables[len(perm):])
    else:
        permuted_variables = variables
    
    # Build expression based on tree layout
    if assembler.tree_layout == "full" and assembler.fully_connected_tree is not None:
        # Full tree: use the FullyConnectedTree structure with permuted variables
        return _reconstruct_full_tree(assembler.fully_connected_tree, permuted_variables, operators,
                                      weight_threshold, show_weights, precision)
    else:
        # Left/Balanced tree: use the standard weights (original variable order)
        weights_list = []
        for w in assembler.weights:
            # Apply sigmoid if weights are in logit space
            w_val = torch.sigmoid(w).detach().cpu().numpy()
            weights_list.append(w_val)
        
        if assembler.tree_layout == "left":
            return _reconstruct_left_tree(variables, operators, weights_list, 
                                          weight_threshold, show_weights, precision)
        elif assembler.tree_layout == "balanced":
            return _reconstruct_balanced_tree(variables, operators, weights_list,
                                             weight_threshold, show_weights, precision)
        else:
            return _reconstruct_left_tree(variables, operators, weights_list,
                                          weight_threshold, show_weights, precision)


def get_operator_selections(model: "baconNet") -> List[dict]:
    """
    Get the learned operator selections at each node.
    
    Args:
        model: Trained BACON model with OperatorSetAggregator
        
    Returns:
        List of dicts with operator info per node:
        [{'node': 1, 'operator': 'add', 'confidence': 0.95, 'probs': {'add': 0.95, ...}}, ...]
    """
    aggregator = model.assembler.aggregator
    
    if not hasattr(aggregator, 'op_logits_per_node') or aggregator.op_logits_per_node is None:
        return []
    
    op_names = aggregator.op_names
    selections = []
    
    for i, logits in enumerate(aggregator.op_logits_per_node):
        probs = torch.softmax(logits, dim=0)
        best_idx = torch.argmax(probs).item()
        best_op = op_names[best_idx]
        confidence = probs[best_idx].item()
        
        prob_dict = {op: p.item() for op, p in zip(op_names, probs)}
        
        selections.append({
            'node': i + 1,
            'operator': best_op,
            'confidence': confidence,
            'probs': prob_dict
        })
    
    return selections


def print_operator_selections(model: "baconNet", variables: List[str] = None):
    """
    Print which operators were selected at each node.
    
    Args:
        model: Trained BACON model with OperatorSetAggregator
        variables: List of variable names (optional, for context)
    """
    selections = get_operator_selections(model)
    
    if selections:
        logging.info("\n🔧 Learned Operator Selections:")
        for sel in selections:
            probs_str = ", ".join([f"{op}={p:.2f}" for op, p in sel['probs'].items()])
            logging.info(f"   Node {sel['node']}: {sel['operator']} (conf={sel['confidence']:.2f}) [{probs_str}]")


def print_reconstructed_expression(model: "baconNet", variables: List[str], precision: int = 2):
    """
    Print the reconstructed expression from the trained model.
    
    Args:
        model: Trained BACON model with OperatorSetAggregator
        variables: List of variable names (e.g., ['a', 'b', 'c'])
        precision: Number of decimal places for weights (default: 2)
    """
    expr = simplify_expression(
        reconstruct_expression(model, variables, show_weights=True, precision=precision),
        variables=variables,
    )
    expr_no_weights = simplify_expression(
        reconstruct_expression(model, variables, show_weights=False, precision=precision),
        variables=variables,
    )
    
    logging.info("\n📐 Reconstructed Expression:")
    logging.info(f"   With weights:    {expr}")
    logging.info(f"   Structure only:  {expr_no_weights}")


def simplify_expression(expression: str, variables: List[str] | None = None) -> str:
    """Simplify a reconstructed arithmetic expression for display.

    Falls back to the original expression if SymPy is unavailable or the expression
    is not valid SymPy syntax, which keeps boolean/operator-set displays safe.
    """
    if not expression or sympy is None:
        return expression

    try:
        symbol_names = variables or sorted(set(_SYMBOL_PATTERN.findall(expression)))
        local_symbols = {name: sympy.Symbol(name) for name in symbol_names}
        sym_expr = sympy.sympify(expression, locals=local_symbols)
        simplified = sympy.simplify(sym_expr)
        return str(simplified)
    except Exception:
        return expression


# =============================================================================
# Helper functions
# =============================================================================

def _format_weight(weight: float, precision: int = 2) -> str:
    """Format a weight value for display."""
    if abs(weight - 1.0) < 0.01:
        return ""
    elif abs(weight - round(weight)) < 0.01:
        return f"{int(round(weight))}*"
    else:
        return f"{weight:.{precision}f}*"


def _format_term(var: str, weight: float, show_weights: bool, precision: int) -> str:
    """Format a single term (weight * variable)."""
    if show_weights:
        w_str = _format_weight(weight, precision)
        return f"{w_str}{var}"
    return var


def _op_symbol(op: str) -> str:
    """Convert operator name to symbol."""
    symbols = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
        'and': '∧',
        'or': '∨',
        'identity': '→',
        'zero': '0',
    }
    return symbols.get(op.lower(), op)


def _reconstruct_left_tree(
    variables: List[str],
    operators: List[str],
    weights_list: List[np.ndarray],
    weight_threshold: float,
    show_weights: bool,
    precision: int
) -> str:
    """
    Reconstruct expression from a left-associative tree.
    
    Tree structure: ((var[0] op var[1]) op var[2]) op var[3] ...
    Each node i combines the result so far with var[i+1]
    """
    if len(variables) == 0:
        return ""
    if len(variables) == 1:
        return variables[0]
    
    # Build expression iteratively
    # First node combines var[0] and var[1]
    # weights_list[i] has [w_left, w_right] for node i
    
    if len(weights_list) > 0 and len(weights_list[0]) >= 2:
        w0 = float(weights_list[0][0])
        w1 = float(weights_list[0][1])
    else:
        w0, w1 = 1.0, 1.0
    
    op = operators[0] if len(operators) > 0 else 'add'
    
    # Format first two terms
    t0 = _format_term(variables[0], w0, show_weights, precision)
    t1 = _format_term(variables[1], w1, show_weights, precision)
    
    if op == 'sub':
        expr = f"({t0} - {t1})"
    elif op == 'mul':
        expr = f"({t0} * {t1})"
    elif op == 'div':
        expr = f"({t0} / {t1})"
    else:  # add is default
        expr = f"({t0} + {t1})"
    
    # Add remaining variables
    for i in range(2, len(variables)):
        node_idx = i - 1  # Node index for this combination
        
        if node_idx < len(weights_list) and len(weights_list[node_idx]) >= 2:
            w_left = float(weights_list[node_idx][0])
            w_right = float(weights_list[node_idx][1])
        else:
            w_left, w_right = 1.0, 1.0
        
        op = operators[node_idx] if node_idx < len(operators) else 'add'
        
        var_term = _format_term(variables[i], w_right, show_weights, precision)
        
        # Apply weight to accumulated expression if needed
        if show_weights and abs(w_left - 1.0) > 0.01:
            left_weight = _format_weight(w_left, precision)
            expr = f"{left_weight}{expr}"
        
        if op == 'sub':
            expr = f"({expr} - {var_term})"
        elif op == 'mul':
            expr = f"({expr} * {var_term})"
        elif op == 'div':
            expr = f"({expr} / {var_term})"
        else:
            expr = f"({expr} + {var_term})"
    
    return expr


def _reconstruct_balanced_tree(
    variables: List[str],
    operators: List[str],
    weights_list: List[np.ndarray],
    weight_threshold: float,
    show_weights: bool,
    precision: int
) -> str:
    """
    Reconstruct expression from a balanced tree.
    
    For balanced trees, variables are paired bottom-up.
    """
    if len(variables) == 0:
        return ""
    if len(variables) == 1:
        return variables[0]
    
    # For simplicity, use the same left-tree reconstruction
    # A more accurate implementation would trace the actual tree structure
    return _reconstruct_left_tree(variables, operators, weights_list,
                                  weight_threshold, show_weights, precision)


def _reconstruct_full_tree(
    full_tree,
    variables: List[str],
    operators: List[str],
    weight_threshold: float,
    show_weights: bool,
    precision: int
) -> str:
    """
    Reconstruct expression from a fully connected tree.
    
    This traces the edges through each layer to build a composition of operators.
    The tree has:
    - Layer 0: input variables
    - Layer 1...depth: internal nodes with operators
    
    Each internal node applies an operator to its weighted inputs.
    """
    structure = full_tree.get_tree_structure()
    edges = structure['edges']
    layer_widths = structure['layer_widths']
    depth = structure['depth']
    
    if len(variables) == 0:
        return ""
    if len(variables) == 1:
        return variables[0]
    
    # Build a representation for each node at each layer
    # Layer 0 nodes are the input variables
    node_exprs = {0: {i: variables[i] if i < len(variables) else f"x{i}" 
                      for i in range(layer_widths[0])}}
    
    # Organize edges by layer and destination
    # Use combined weight = selection_weight * scale for display
    edges_by_layer_dst = {}
    for edge in edges:
        layer = edge['layer']
        dst = edge['dst']
        select_weight = edge['weight']
        scale = edge.get('scale', 1.0)  # Default to 1.0 if no scale
        combined_weight = select_weight * scale
        
        if layer not in edges_by_layer_dst:
            edges_by_layer_dst[layer] = {}
        if dst not in edges_by_layer_dst[layer]:
            edges_by_layer_dst[layer][dst] = []
        edges_by_layer_dst[layer][dst].append((edge['src'], combined_weight, select_weight))
    
    # Process each layer
    op_idx = 0
    for layer in range(depth):
        node_exprs[layer + 1] = {}
        out_width = layer_widths[layer + 1]
        
        for dst in range(out_width):
            # Get incoming edges for this node
            incoming = edges_by_layer_dst.get(layer, {}).get(dst, [])
            
            if len(incoming) == 0:
                # No significant edges - use a placeholder
                node_exprs[layer + 1][dst] = "0"
                continue
            
            # Sort by combined weight (descending) to put more important terms first
            incoming = sorted(incoming, key=lambda x: -abs(x[1]))
            
            # Get operator for this node
            op = operators[op_idx] if op_idx < len(operators) else 'add'
            op_idx += 1
            
            # Build expression for this node
            terms = []
            for src, combined_weight, select_weight in incoming:
                # Skip if selection weight is below threshold (edge not active)
                if select_weight < weight_threshold:
                    continue
                src_expr = node_exprs[layer].get(src, f"?{src}")
                
                if show_weights and abs(combined_weight - 1.0) > 0.05:
                    w_str = _format_weight(combined_weight, precision)
                    terms.append(f"{w_str}{src_expr}")
                else:
                    terms.append(src_expr)
            
            if len(terms) == 0:
                node_exprs[layer + 1][dst] = "0"
            elif len(terms) == 1:
                node_exprs[layer + 1][dst] = terms[0]
            else:
                # Combine terms with operator
                op_sym = _op_symbol(op)
                if op == 'add':
                    combined = " + ".join(terms)
                elif op == 'sub':
                    combined = f"{terms[0]} - " + " - ".join(terms[1:])
                elif op == 'mul':
                    combined = " * ".join(terms)
                elif op == 'div':
                    combined = f"{terms[0]} / " + " / ".join(terms[1:])
                elif op == 'identity':
                    # Identity: just use the first term
                    combined = terms[0]
                elif op == 'zero':
                    # Zero: output is always 0
                    combined = "0"
                else:
                    combined = f" {op_sym} ".join(terms)
                
                node_exprs[layer + 1][dst] = f"({combined})"
    
    # The output is at the final layer, node 0
    final_layer = depth
    if final_layer in node_exprs and 0 in node_exprs[final_layer]:
        return node_exprs[final_layer][0]
    return "?"
