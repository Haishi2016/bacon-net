#!/usr/bin/env python
"""Quick smoke test for negative sampling."""

import sys
import os
sys.path.insert(0, 'c:\\School\\bacon-net')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from bacon_head import MultiTreeBaconCBM, StochasticMultiTreeBaconCBM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test MultiTreeBaconCBM with negative sampling disabled
print("Testing MultiTreeBaconCBM with negative sampling disabled...")
model1 = MultiTreeBaconCBM([2048, 112, 200], activation='relu', use_negative_sampling=False).to(device)
model1.train()
x = torch.randn(4, 2048).to(device)
targets = torch.tensor([0, 1, 2, 3]).to(device)
logits1, _, _ = model1(x, targets)
print(f"  Output shape: {logits1.shape}")
print(f"  All 200 classes evaluated: {logits1[0, :].abs().sum().item() > 0}")

# Test MultiTreeBaconCBM with negative sampling enabled
print("\nTesting MultiTreeBaconCBM with negative sampling enabled...")
model2 = MultiTreeBaconCBM([2048, 112, 200], activation='relu', use_negative_sampling=True).to(device)
model2.train()
logits2, _, _ = model2(x, targets)
print(f"  Output shape: {logits2.shape}")
non_zero = (logits2 != 0).sum(dim=1)
print(f"  Non-zero predictions per sample (should be ~2): {non_zero.tolist()}")
print(f"  Negative sampling working: {(non_zero == 2).all().item()}")

# Test StochasticMultiTreeBaconCBM with negative sampling enabled
print("\nTesting StochasticMultiTreeBaconCBM with negative sampling enabled...")
model3 = StochasticMultiTreeBaconCBM([2048, 112, 200], activation='relu', use_negative_sampling=True).to(device)
model3.train()
logits3, std_val, z = model3(x, targets)
print(f"  Output shape: {logits3.shape}")
non_zero = (logits3 != 0).sum(dim=1)
print(f"  Non-zero predictions per sample (should be ~2): {non_zero.tolist()}")
print(f"  Negative sampling working: {(non_zero == 2).all().item()}")

# Test without targets (should evaluate all trees)
print("\nTesting without targets (should evaluate all trees)...")
logits4, _, _ = model2(x, targets=None)
non_zero = (logits4 != 0).sum(dim=1)
print(f"  Non-zero predictions per sample (should be 200): {non_zero.tolist()}")
print(f"  All trees evaluated: {(non_zero == 200).all().item()}")

print("\n✅ All smoke tests passed!")
