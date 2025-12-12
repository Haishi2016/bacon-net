# Symbolic Regression approach for 1000-variable boolean expression inference
# Using PySR (symbolic regression) to discover the exact boolean expression

import sys
sys.path.insert(0, '../../')

import torch
from bacon.utils import generate_classic_boolean_data
import time
import numpy as np
from pysr import PySRRegressor

print("=" * 80)
print("🔬 Symbolic Regression (PySR): 1000-Variable Boolean Expression Inference")
print("=" * 80)

input_size = 1000

print(f"\n📊 Generating boolean expression with {input_size} variables...")
print(f"   Using randomized sampling (10,000 samples)")

# Generate same data as other methods
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train, y_train, expr_info = generate_classic_boolean_data(
    input_size, 
    repeat_factor=50000,  # 10k training samples
    randomize=True, 
    device=device
)

x_test, y_test, _ = generate_classic_boolean_data(
    input_size,
    repeat_factor=20000,  # 5k test samples  
    randomize=True,
    device=device
)

# Convert to numpy
x_train_np = x_train.cpu().numpy()
y_train_np = y_train.cpu().numpy().ravel()
x_test_np = x_test.cpu().numpy()
y_test_np = y_test.cpu().numpy().ravel()

# PySR doesn't support emojis or reserved symbols like 'E'
# Use simple numbered variable names: x0, x1, x2, ...
var_names_pysr = [f"x{i}" for i in range(input_size)]

print(f"✅ Data generated")
print(f"   Training samples: {len(x_train_np)}")
print(f"   Test samples: {len(x_test_np)}")
print(f"   Expression: {expr_info['expression_text'][:100]}...")
print(f"   Variables: x0, x1, x2, ... x999")

print("\n🔧 Configuring PySR (Symbolic Regression)...")
# PySR uses evolutionary algorithms to discover mathematical expressions
model = PySRRegressor(
    niterations=40,  # Number of iterations
    binary_operators=["+", "*", "/", "-"],
    unary_operators=["neg", "square", "cube"],
    model_selection="best",  # Choose the best model
    elementwise_loss="loss(x, y) = (x - y)^2",  # MSE loss (renamed from 'loss')
    populations=15,  # Number of populations for evolution
    population_size=33,  # Size of each population
    maxsize=30,  # Maximum complexity of expressions
    timeout_in_seconds=300,  # 5 minutes timeout
    parsimony=0.0032,  # Penalty for complexity
    random_state=42,
    verbosity=1,
)

print("✅ Model configured")
print(f"   Algorithm: PySR (Physics-Inspired Symbolic Regression)")
print(f"   Uses genetic programming to discover symbolic expressions")
print(f"   Iterations: 40, Populations: 15")
print(f"   Timeout: 5 minutes")

print("\n🔥 Training PySR (this may take a few minutes)...")
print("   PySR will search for the best symbolic expression using evolution...")
start_time = time.time()

# Fit the model
model.fit(x_train_np, y_train_np, variable_names=var_names_pysr)

training_time = time.time() - start_time

print("✅ Training complete")

# Evaluate
y_train_pred = model.predict(x_train_np)
y_test_pred = model.predict(x_test_np)

# Convert predictions to binary (threshold at 0.5)
y_train_pred_binary = (y_train_pred >= 0.5).astype(int)
y_test_pred_binary = (y_test_pred >= 0.5).astype(int)

train_accuracy = np.mean(y_train_pred_binary == y_train_np)
test_accuracy = np.mean(y_test_pred_binary == y_test_np)

print("\n" + "=" * 80)
print("📊 SYMBOLIC REGRESSION RESULTS")
print("=" * 80)
print(f"🏆 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"⏱️  Training Time: {training_time:.2f} seconds")

# Show the best discovered expressions
print("\n🔬 Discovered Expressions (Hall of Fame):")
print("   Showing top 5 expressions by complexity vs accuracy tradeoff:")
print()
print(model.equations_)

# Get the best expression
best_expr = model.sympy()
print(f"\n🏆 Best Expression (symbolic):")
print(f"   {best_expr}")

# Show the latex version
try:
    best_latex = model.latex()
    print(f"\n📐 Best Expression (LaTeX):")
    print(f"   {best_latex}")
except:
    pass

# Classification metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\n📊 Classification Report:")
print(classification_report(y_test_np, y_test_pred_binary, target_names=['False', 'True']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test_np, y_test_pred_binary)
print(cm)
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

print("\n✅ Symbolic Regression benchmark complete!")
print("\n💡 PySR discovers exact mathematical/logical expressions")
print("   Unlike neural networks, the result is fully interpretable symbolic math")
