"""
Distilled BACON Network - Standalone Inference Code (Instance Mode)
Generated automatically from trained BACON model.

This file is self-contained and can perform inference without any dependencies.
Zero dependencies: Uses only Python standard library (math module).
"""

import math


# ============================================================================
# Aggregator Functions
# ============================================================================


def lsp_half_weight_r(a):
    """Compute r parameter for LSP half-weight aggregator."""
    delta = 0.5 - a
    numerator = (0.25 +
                 1.65811 * delta +
                 2.15388 * delta ** 2 + 
                 8.2844 * delta ** 3 +
                 6.16764 * delta ** 4)
    denominator = a * (1 - a)
    epsilon = 1e-6
    if abs(denominator) < epsilon:
        denominator = epsilon if denominator >= 0 else -epsilon
    return numerator / denominator


def lsp_half_weight_aggregate(x, y, a, w0, w1):
    """LSP half-weight aggregator for two inputs.
    
    Args:
        x: First input (0 to 1)
        y: Second input (0 to 1)
        a: Andness parameter (-1 to 2)
        w0: Weight for first input
        w1: Weight for second input
        
    Returns:
        Aggregated value (0 to 1)
    """
    epsilon = 1e-6
    
    # Clamp inputs to valid range
    x = max(epsilon, min(1 - epsilon, x))
    y = max(epsilon, min(1 - epsilon, y))
    a = max(-1.0 + epsilon, min(2.0 - epsilon, a))
    
    # Rule 0: a == 2 (full conjunction)
    if abs(a - 2) < epsilon:
        return 1.0 if (abs(x - 1) < epsilon and abs(y - 1) < epsilon) else 0.0
    
    # Rule 1: 0.75 <= a < 2 (strong conjunction)
    elif a >= 0.75:
        import math
        return (x ** (2*w0) * y ** (2*w1)) ** (math.sqrt(3 / (2 - a)) - 1)
    
    # Rule 2: 0.5 < a < 0.75 (weak conjunction)
    elif a > 0.5:
        import math
        R = lsp_half_weight_r(0.75)
        return (3 - 4*a) * (w0*x + w1*y) + (4*a - 2) * (x ** (2*w0) * y ** (2*w1)) ** (math.sqrt(3 / (2 - a)) - 1)
    
    # Rule 3: a == 0.5 (arithmetic mean)
    elif abs(a - 0.5) < epsilon:
        return w0*x + w1*y
    
    # Rule 4: -1 <= a < 0.5 (disjunction, use De Morgan)
    elif a >= -1:
        return 1 - lsp_half_weight_aggregate(1-x, 1-y, max(-1.0 + epsilon, min(2.0 - epsilon, 1-a)), w0, w1)
    
    else:
        raise ValueError(f"Invalid andness value: {a}. Must be in [-1, 2].")


# ============================================================================
# Transformation Functions
# ============================================================================


def apply_identity(x):
    """Identity transformation."""
    return x


def apply_negation(x):
    """Negation transformation."""
    return 1.0 - x


def apply_peak(x, center=0.5, sharpness=2.0):
    """Peak transformation - bell curve centered at 'center'."""
    return 1.0 - abs(x - center) ** sharpness


def apply_valley(x, center=0.5, sharpness=2.0):
    """Valley transformation - inverted bell curve."""
    return abs(x - center) ** sharpness


def apply_step_up(x, threshold=0.5, sharpness=10.0):
    """Step up transformation - sigmoid-like step."""
    import math
    return 1.0 / (1.0 + math.exp(-sharpness * (x - threshold)))


def apply_step_down(x, threshold=0.5, sharpness=10.0):
    """Step down transformation - inverted sigmoid."""
    import math
    return 1.0 / (1.0 + math.exp(sharpness * (x - threshold)))


# ============================================================================
# Model Inference
# ============================================================================


def predict(input_array):
    """Perform inference on input data.
    
    Args:
        input_array: List or array of 22 input features in ORIGINAL dataset order
                     Feature order: ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'restecg_0', 'restecg_1', 'restecg_2', 'slope_1', 'slope_2', 'slope_3', 'thal_3.0', 'thal_6.0', 'thal_7.0']
        
    Returns:
        float: Prediction value (0 to 1)
    """
    if len(input_array) != 22:
        raise ValueError(f"Expected 22 features, got {len(input_array)}")
    
    # Apply permutation and transformations
    features = []

    features.append(apply_identity(input_array[11]))  # cp_3
    features.append(apply_identity(input_array[9]))  # cp_1
    features.append(apply_identity(input_array[19]))  # thal_3.0
    features.append(apply_identity(input_array[15]))  # restecg_2
    features.append(apply_identity(input_array[4]))  # fbs
    features.append(apply_identity(input_array[20]))  # thal_6.0
    features.append(apply_identity(input_array[5]))  # thalach
    features.append(apply_identity(input_array[3]))  # chol
    features.append(apply_identity(input_array[13]))  # restecg_0
    features.append(apply_identity(input_array[16]))  # slope_1
    features.append(apply_identity(input_array[14]))  # restecg_1
    features.append(apply_identity(input_array[10]))  # cp_2
    features.append(apply_identity(input_array[2]))  # trestbps
    features.append(apply_identity(input_array[18]))  # slope_3
    features.append(apply_identity(input_array[1]))  # sex
    features.append(apply_identity(input_array[0]))  # age
    features.append(apply_identity(input_array[8]))  # ca
    features.append(apply_identity(input_array[7]))  # oldpeak
    features.append(apply_identity(input_array[17]))  # slope_2
    features.append(apply_identity(input_array[6]))  # exang
    features.append(apply_identity(input_array[21]))  # thal_7.0
    features.append(apply_identity(input_array[12]))  # cp_4

    # Aggregate through the tree

    agg_0 = lsp_half_weight_aggregate(features[0], features[1], 0.827083, 0.5, 0.5)  # Layer 0
    agg_1 = lsp_half_weight_aggregate(agg_0, features[2], -0.513749, 0.5, 0.5)  # Layer 1
    agg_2 = lsp_half_weight_aggregate(agg_1, features[3], -0.625657, 0.5, 0.5)  # Layer 2
    agg_3 = lsp_half_weight_aggregate(agg_2, features[4], -0.254279, 0.5, 0.5)  # Layer 3
    agg_4 = lsp_half_weight_aggregate(agg_3, features[5], -0.083511, 0.5, 0.5)  # Layer 4
    agg_5 = lsp_half_weight_aggregate(agg_4, features[6], 0.520075, 0.5, 0.5)  # Layer 5
    agg_6 = lsp_half_weight_aggregate(agg_5, features[7], -0.565187, 0.5, 0.5)  # Layer 6
    agg_7 = lsp_half_weight_aggregate(agg_6, features[8], 0.654859, 0.5, 0.5)  # Layer 7
    agg_8 = lsp_half_weight_aggregate(agg_7, features[9], 0.581074, 0.5, 0.5)  # Layer 8
    agg_9 = lsp_half_weight_aggregate(agg_8, features[10], -0.084794, 0.5, 0.5)  # Layer 9
    agg_10 = lsp_half_weight_aggregate(agg_9, features[11], -0.122285, 0.5, 0.5)  # Layer 10
    agg_11 = lsp_half_weight_aggregate(agg_10, features[12], 0.202179, 0.5, 0.5)  # Layer 11
    agg_12 = lsp_half_weight_aggregate(agg_11, features[13], -0.734053, 0.5, 0.5)  # Layer 12
    agg_13 = lsp_half_weight_aggregate(agg_12, features[14], 1.716753, 0.5, 0.5)  # Layer 13
    agg_14 = lsp_half_weight_aggregate(agg_13, features[15], -0.226167, 0.5, 0.5)  # Layer 14
    agg_15 = lsp_half_weight_aggregate(agg_14, features[16], 1.743444, 0.5, 0.5)  # Layer 15
    agg_16 = lsp_half_weight_aggregate(agg_15, features[17], 1.622845, 0.5, 0.5)  # Layer 16
    agg_17 = lsp_half_weight_aggregate(agg_16, features[18], 1.468585, 0.5, 0.5)  # Layer 17
    agg_18 = lsp_half_weight_aggregate(agg_17, features[19], 0.750886, 0.5, 0.5)  # Layer 18
    agg_19 = lsp_half_weight_aggregate(agg_18, features[20], 0.536983, 0.5, 0.5)  # Layer 19
    agg_20 = lsp_half_weight_aggregate(agg_19, features[21], 1.254692, 0.5, 0.5)  # Layer 20

    return agg_20



if __name__ == "__main__":
    # Example usage for instance mode
    import sys
    
    if len(sys.argv) > 1:
        # Read input from command line
        try:
            input_values = [float(x) for x in sys.argv[1:]]
            result = predict(input_values)
            print(f"Prediction: {result:.6f}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Usage: python {sys.argv[0]} <value1> <value2> ... <valueN>")
    else:
        # Demo with random input
        import random
        demo_input = [random.random() for _ in range(22)]
        print(f"Demo input: {demo_input}")
        result = predict(demo_input)
        print(f"Prediction: {result:.6f}")
