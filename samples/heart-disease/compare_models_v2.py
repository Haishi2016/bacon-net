"""
Compare Original BACON Model vs Distilled Models (Instance and Batch Modes)

This script validates that both distilled models produce identical predictions
to the original BACON model, and measures the performance of each mode.
"""

import sys
sys.path.insert(0, '../../')

import torch
import time
import numpy as np
from dataset import prepare_data

# Import both distilled models
try:
    from heart_disease_model_instance import predict as instance_predict
    has_instance = True
except ImportError:
    print("⚠️  Warning: heart_disease_model_instance.py not found!")
    has_instance = False

try:
    from heart_disease_model_batch import predict as batch_predict
    has_batch = True
except ImportError:
    print("⚠️  Warning: heart_disease_model_batch.py not found!")
    has_batch = False

if not has_instance and not has_batch:
    print("\n❌ Error: No distilled models found!")
    print("Please run distillation first:")
    print("  Instance mode: python ../../bacon-distill.py heart_disease_tree_structure.json heart_disease_model_instance.py --mode instance")
    print("  Batch mode:    python ../../bacon-distill.py heart_disease_tree_structure.json heart_disease_model_batch.py --mode batch")
    sys.exit(1)

# Import original model
from bacon.baconNet import baconNet

print("="*80)
print("BACON MODEL COMPARISON: Original vs Distilled (Instance & Batch)")
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
            bacon.assembler.is_frozen = checkpoint.get('is_frozen', False)
            bacon.assembler.locked_perm = checkpoint.get('locked_perm', None)
            bacon.assembler.tree_layout = checkpoint.get('tree_layout', 'left')
            
            # If model was frozen, recreate frozen input layer
            if bacon.assembler.is_frozen and bacon.assembler.locked_perm is not None:
                from bacon.frozonInputToLeaf import frozenInputToLeaf
                bacon.assembler.input_to_leaf = frozenInputToLeaf(
                    bacon.assembler.locked_perm, 
                    bacon.assembler.original_input_size
                ).to(device)
            
            bacon.assembler.load_state_dict(checkpoint['model_state_dict'])
        else:
            bacon.assembler.load_state_dict(checkpoint)
        
        bacon.eval()
        print("   ✅ Model loaded successfully")
    else:
        print("   ⚠️  No saved model found.")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("PREDICTION ACCURACY TEST")
print("="*80)

# Convert data to numpy for distilled models
X_all_np = X_all.cpu().numpy()
Y_all_np = Y_all.cpu().numpy().flatten()

# Get predictions from original model
print("\n   Running original model (batch)...")
with torch.no_grad():
    original_predictions = bacon(X_all).cpu().numpy().flatten()

tolerance = 1e-6

# Test instance mode if available
if has_instance:
    print("   Running instance mode distilled model...")
    instance_predictions = np.array([instance_predict(X_all_np[i]) for i in range(num_samples)])
    
    diff_instance = np.abs(original_predictions - instance_predictions)
    max_diff_instance = np.max(diff_instance)
    mean_diff_instance = np.mean(diff_instance)
    
    print(f"\n📈 Instance Mode - Prediction Differences:")
    print(f"   Maximum difference: {max_diff_instance:.10f}")
    print(f"   Mean difference:    {mean_diff_instance:.10f}")
    
    if max_diff_instance < tolerance:
        print(f"   ✅ PASSED: Instance predictions identical (within {tolerance})")
    else:
        print(f"   ⚠️  WARNING: Predictions differ by more than {tolerance}")

# Test batch mode if available
if has_batch:
    print("\n   Running batch mode distilled model...")
    batch_predictions = batch_predict(X_all_np).flatten()
    
    diff_batch = np.abs(original_predictions - batch_predictions)
    max_diff_batch = np.max(diff_batch)
    mean_diff_batch = np.mean(diff_batch)
    
    print(f"\n📈 Batch Mode - Prediction Differences:")
    print(f"   Maximum difference: {max_diff_batch:.10f}")
    print(f"   Mean difference:    {mean_diff_batch:.10f}")
    
    if max_diff_batch < tolerance:
        print(f"   ✅ PASSED: Batch predictions identical (within {tolerance})")
    else:
        print(f"   ⚠️  WARNING: Predictions differ by more than {tolerance}")

# Compare instance and batch modes
if has_instance and has_batch:
    diff_modes = np.abs(instance_predictions - batch_predictions)
    max_diff_modes = np.max(diff_modes)
    print(f"\n📈 Instance vs Batch Mode:")
    print(f"   Maximum difference: {max_diff_modes:.10f}")
    if max_diff_modes < tolerance:
        print(f"   ✅ Both modes produce identical results")

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

# Warmup and benchmark original model
print("\n🔥 Warming up original model...")
for _ in range(num_warmup):
    with torch.no_grad():
        _ = bacon(X_all)

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

print(f"\n📊 Original Model (Batch):")
print(f"   Total time:    {original_mean*1000:.2f} ± {original_std*1000:.2f} ms")
print(f"   Per sample:    {original_mean*1000/num_samples:.4f} ms")
print(f"   Throughput:    {num_samples/original_mean:.0f} samples/sec")

# Benchmark instance mode
if has_instance:
    print("\n🔥 Warming up instance mode...")
    for _ in range(num_warmup):
        for i in range(num_samples):
            _ = instance_predict(X_all_np[i])
    
    print("⏱️  Benchmarking instance mode...")
    instance_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        for i in range(num_samples):
            _ = instance_predict(X_all_np[i])
        end = time.perf_counter()
        instance_times.append(end - start)
    
    instance_mean = np.mean(instance_times)
    instance_std = np.std(instance_times)
    instance_speedup = original_mean / instance_mean
    
    print(f"\n📊 Instance Mode (Per-sample):")
    print(f"   Total time:    {instance_mean*1000:.2f} ± {instance_std*1000:.2f} ms")
    print(f"   Per sample:    {instance_mean*1000/num_samples:.4f} ms")
    print(f"   Throughput:    {num_samples/instance_mean:.0f} samples/sec")
    print(f"   🚀 Speedup:     {instance_speedup:.2f}x vs original")
    print(f"   📦 Dependencies: None (zero deps)")

# Benchmark batch mode
if has_batch:
    print("\n🔥 Warming up batch mode...")
    for _ in range(num_warmup):
        _ = batch_predict(X_all_np)
    
    print("⏱️  Benchmarking batch mode...")
    batch_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = batch_predict(X_all_np)
        end = time.perf_counter()
        batch_times.append(end - start)
    
    batch_mean = np.mean(batch_times)
    batch_std = np.std(batch_times)
    batch_speedup = original_mean / batch_mean
    
    print(f"\n📊 Batch Mode (Vectorized):")
    print(f"   Total time:    {batch_mean*1000:.2f} ± {batch_std*1000:.2f} ms")
    print(f"   Per sample:    {batch_mean*1000/num_samples:.4f} ms")
    print(f"   Throughput:    {num_samples/batch_mean:.0f} samples/sec")
    print(f"   🚀 Speedup:     {batch_speedup:.2f}x vs original")
    print(f"   📦 Dependencies: NumPy")

# Compare modes
if has_instance and has_batch:
    mode_ratio = instance_mean / batch_mean
    print(f"\n📊 Mode Comparison:")
    print(f"   Batch is {mode_ratio:.2f}x faster than instance mode")
    print(f"   💡 Use instance mode for: edge devices, zero-dependency deploys")
    print(f"   💡 Use batch mode for: servers, high-throughput applications")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ Validation Results:")
if has_instance:
    print(f"   • Instance predictions match: {'YES' if max_diff_instance < tolerance else 'NO'}")
if has_batch:
    print(f"   • Batch predictions match: {'YES' if max_diff_batch < tolerance else 'NO'}")
if has_instance and has_batch:
    print(f"   • Modes agree with each other: {'YES' if max_diff_modes < tolerance else 'NO'}")

print(f"\n⚡ Performance Summary:")
print(f"   • Original (PyTorch):  {num_samples/original_mean:.0f} samples/sec")
if has_instance:
    print(f"   • Instance (zero deps): {num_samples/instance_mean:.0f} samples/sec ({instance_speedup:.2f}x)")
if has_batch:
    print(f"   • Batch (NumPy):       {num_samples/batch_mean:.0f} samples/sec ({batch_speedup:.2f}x)")

print(f"\n💡 Deployment Recommendations:")
if has_batch and batch_speedup > 5:
    print(f"   🏆 BEST: Batch mode for production (fastest, {batch_speedup:.1f}x speedup)")
elif has_instance:
    print(f"   🏆 BEST: Instance mode for edge/embedded (zero dependencies)")

print("\n" + "="*80)
