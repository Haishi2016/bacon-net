"""
Compare Original BACON Model vs Distilled Model

This script validates that the distilled model produces identical predictions
to the original BACON model, and measures the performance improvement.
"""

import sys
sys.path.insert(0, '../../')

import torch
import time
import numpy as np
from dataset import prepare_data

# Import distilled model
try:
    from heart_disease_distilled import predict as distilled_predict
except ImportError:
    print("❌ Error: heart_disease_distilled.py not found!")
    print("Please run distillation first:")
    print("  python -m bacon.distill heart_disease_tree_structure.json heart_disease_distilled.py")
    sys.exit(1)

# Import original model
from bacon.baconNet import baconNet

print("="*80)
print("BACON MODEL COMPARISON: Original vs Distilled")
print("="*80)

# Prepare data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)

# Combine all data for comprehensive testing
X_all = torch.cat([X_train, X_test], dim=0)
Y_all = torch.cat([Y_train, Y_test], dim=0)

num_samples = X_all.shape[0]
print(f"\n📊 Dataset: {num_samples} samples ({X_train.shape[0]} train + {X_test.shape[0]} test)")
print(f"📊 Features: {len(feature_names)}")

# Load original model
try:
    print("\n🔄 Loading original BACON model...")
    
    # Import transformation types
    from bacon.transformationLayer import IdentityTransformation, NegationTransformation
    
    # Create model with same configuration as training (only 2 transformations)
    bacon = baconNet(
        input_size=len(feature_names),
        aggregator='lsp.half_weight',
        weight_mode='fixed',
        use_transformation_layer=True,
        transformations=[
            IdentityTransformation(1),
            NegationTransformation(1)
        ],
        weight_normalization='softmax',
        use_class_weighting=True
    )
    
    # Try to load from saved state
    import os
    model_files = [f for f in os.listdir('.') if f.startswith('assembler') and f.endswith('.pth')]
    if model_files:
        model_file = model_files[0]
        print(f"   Loading from: {model_file}")
        
        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format: checkpoint with metadata
            # Restore metadata FIRST
            bacon.assembler.is_frozen = checkpoint.get('is_frozen', False)
            bacon.assembler.locked_perm = checkpoint.get('locked_perm', None)
            bacon.assembler.tree_layout = checkpoint.get('tree_layout', 'left')
            
            # If model was frozen, recreate frozen input layer BEFORE loading state_dict
            if bacon.assembler.is_frozen and bacon.assembler.locked_perm is not None:
                from bacon.frozonInputToLeaf import frozenInputToLeaf
                bacon.assembler.input_to_leaf = frozenInputToLeaf(
                    bacon.assembler.locked_perm, 
                    bacon.assembler.original_input_size
                ).to(device)
            
            # Now load model state
            bacon.assembler.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Old format: direct state dict
            bacon.assembler.load_state_dict(checkpoint)
        
        bacon.eval()
        print("   ✅ Model loaded successfully")
    else:
        print("   ⚠️  No saved model found. You may need to train first.")
        print("   Continuing with untrained model for comparison...")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    print("   Please ensure you have a trained model saved.")
    sys.exit(1)

print("\n" + "="*80)
print("PREDICTION ACCURACY TEST")
print("="*80)

# Test predictions match
print("\n🔬 Testing prediction equivalence...")

# Convert data to numpy for distilled model
X_all_np = X_all.cpu().numpy()
Y_all_np = Y_all.cpu().numpy().flatten()

# DEBUG: Check if model has permutation
if bacon.assembler.locked_perm is not None:
    print(f"\n🔍 Model has locked permutation: {bacon.assembler.locked_perm.tolist()}")
    print(f"   This means the model expects features in ORIGINAL order")
    print(f"   The model will internally permute them using locked_perm")
else:
    print("\n🔍 Model has no permutation (features in original order)")

# Get predictions from both models
print("\n   Running original model...")
with torch.no_grad():
    original_predictions = bacon(X_all).cpu().numpy().flatten()

print("   Running distilled model...")

# Test with first sample to debug
print("\n🔍 Debugging first sample:")
test_sample = X_all_np[0]
print(f"   Input shape: {test_sample.shape}")
print(f"   First 5 features: {test_sample[:5]}")
distilled_test = distilled_predict(test_sample)
print(f"   Distilled output: {distilled_test:.8f}")
print(f"   Original output:  {original_predictions[0]:.8f}")
print(f"   Difference: {abs(distilled_test - original_predictions[0]):.8f}")

distilled_predictions = np.array([distilled_predict(X_all_np[i]) for i in range(num_samples)])

# Compare predictions
diff = np.abs(original_predictions - distilled_predictions)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
median_diff = np.median(diff)

print(f"\n📈 Prediction Differences:")
print(f"   Maximum difference: {max_diff:.10f}")
print(f"   Mean difference:    {mean_diff:.10f}")
print(f"   Median difference:  {median_diff:.10f}")

# Check if predictions are essentially identical (within floating point tolerance)
tolerance = 1e-6
if max_diff < tolerance:
    print(f"\n✅ PASSED: Predictions are identical (within tolerance {tolerance})")
else:
    print(f"\n⚠️  WARNING: Predictions differ by more than {tolerance}")
    print(f"   This may indicate an issue with distillation or floating point precision")
    
    # Show some examples of differences
    print("\n   Top 5 largest differences:")
    top_diff_indices = np.argsort(diff)[-5:][::-1]
    for idx in top_diff_indices:
        print(f"   Sample {idx}: Original={original_predictions[idx]:.8f}, "
              f"Distilled={distilled_predictions[idx]:.8f}, "
              f"Diff={diff[idx]:.8f}")

# Classification accuracy comparison (using 0.5 threshold)
threshold = 0.5
original_classes = (original_predictions > threshold).astype(int)
distilled_classes = (distilled_predictions > threshold).astype(int)

original_accuracy = np.mean(original_classes == Y_all_np)
distilled_accuracy = np.mean(distilled_classes == Y_all_np)

print(f"\n📊 Classification Accuracy (threshold={threshold}):")
print(f"   Original model:  {original_accuracy*100:.2f}%")
print(f"   Distilled model: {distilled_accuracy*100:.2f}%")

if original_accuracy == distilled_accuracy:
    print("   ✅ Accuracies match exactly")
else:
    print(f"   ⚠️  Accuracy difference: {abs(original_accuracy - distilled_accuracy)*100:.4f}%")

print("\n" + "="*80)
print("PERFORMANCE BENCHMARK")
print("="*80)

# Benchmark parameters
num_warmup = 10
num_iterations = 100

print(f"\n⏱️  Benchmark settings:")
print(f"   Warmup iterations: {num_warmup}")
print(f"   Test iterations:   {num_iterations}")
print(f"   Samples per run:   {num_samples}")

# Warmup original model
print("\n🔥 Warming up original model...")
for _ in range(num_warmup):
    with torch.no_grad():
        _ = bacon(X_all)

# Benchmark original model
print("⏱️  Benchmarking original model...")
original_times = []
for _ in range(num_iterations):
    start = time.perf_counter()
    with torch.no_grad():
        _ = bacon(X_all)
    end = time.perf_counter()
    original_times.append(end - start)

original_mean = np.mean(original_times)
original_std = np.std(original_times)
original_per_sample = original_mean / num_samples

# Warmup distilled model
print("🔥 Warming up distilled model...")
for _ in range(num_warmup):
    for i in range(num_samples):
        _ = distilled_predict(X_all_np[i])

# Benchmark distilled model
print("⏱️  Benchmarking distilled model...")
distilled_times = []
for _ in range(num_iterations):
    start = time.perf_counter()
    for i in range(num_samples):
        _ = distilled_predict(X_all_np[i])
    end = time.perf_counter()
    distilled_times.append(end - start)

distilled_mean = np.mean(distilled_times)
distilled_std = np.std(distilled_times)
distilled_per_sample = distilled_mean / num_samples

# Calculate speedup
speedup = original_mean / distilled_mean

print(f"\n📊 Performance Results:")
print(f"\n   Original Model:")
print(f"      Total time:    {original_mean*1000:.2f} ± {original_std*1000:.2f} ms")
print(f"      Per sample:    {original_per_sample*1000:.4f} ms")
print(f"      Throughput:    {num_samples/original_mean:.0f} samples/sec")

print(f"\n   Distilled Model:")
print(f"      Total time:    {distilled_mean*1000:.2f} ± {distilled_std*1000:.2f} ms")
print(f"      Per sample:    {distilled_per_sample*1000:.4f} ms")
print(f"      Throughput:    {num_samples/distilled_mean:.0f} samples/sec")

print(f"\n   🚀 Speedup: {speedup:.2f}x faster")

if speedup > 1:
    print(f"   ✅ Distilled model is {speedup:.2f}x faster!")
    time_saved = (original_mean - distilled_mean) * 1000
    print(f"   💰 Time saved per batch: {time_saved:.2f} ms")
else:
    print(f"   ⚠️  Distilled model is slower (may need optimization)")

print("\n" + "="*80)
print("MEMORY FOOTPRINT")
print("="*80)

# Estimate memory usage
import os

# Original model size (approximate)
original_model_params = sum(p.numel() for p in bacon.parameters())
original_model_size = original_model_params * 4  # 4 bytes per float32
print(f"\n📦 Original Model:")
print(f"   Parameters: {original_model_params:,}")
print(f"   Estimated size: {original_model_size / 1024:.2f} KB")

# Distilled model size (file size)
distilled_file = 'heart_disease_distilled.py'
if os.path.exists(distilled_file):
    distilled_size = os.path.getsize(distilled_file)
    print(f"\n📦 Distilled Model:")
    print(f"   File size: {distilled_size / 1024:.2f} KB")
    print(f"   💾 Size reduction: {original_model_size / distilled_size:.2f}x smaller file")

# Framework dependencies
print(f"\n📚 Dependencies:")
print(f"   Original:  PyTorch, NumPy, bacon-net framework")
print(f"   Distilled: None (pure Python + math module)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ Validation Results:")
print(f"   • Predictions match: {'YES' if max_diff < tolerance else 'NO (check tolerance)'}")
print(f"   • Accuracy preserved: {'YES' if original_accuracy == distilled_accuracy else 'NO'}")
print(f"   • Speedup achieved: {speedup:.2f}x")
print(f"   • Zero dependencies: YES")

print(f"\n💡 Conclusion:")
if max_diff < tolerance and speedup > 1:
    print(f"   The distilled model is PRODUCTION READY! 🎉")
    print(f"   • Identical predictions to original")
    print(f"   • {speedup:.2f}x faster inference")
    print(f"   • No external dependencies")
    print(f"   • Ready for deployment to edge devices, serverless, etc.")
else:
    if max_diff >= tolerance:
        print(f"   ⚠️  Predictions differ - review distillation process")
    if speedup <= 1:
        print(f"   ⚠️  Performance not improved - may need optimization")

print("\n" + "="*80)
