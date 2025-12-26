# Understanding Feature Pruning and Accuracy Drops

## The Question

Why does pruning the first feature (Triglyceride) cause a **5% accuracy drop** (from 78.12% to 73.35%) even though:
- It has **balanced weights** (0.5/0.5)
- It has a **very low bias** (a=0.0227, close to 0.5 = arithmetic mean)
- It seems to have "little weight"?

## The Answer: Pruning Changes the Entire Model

### What Actually Happens When You Prune

**Original Model (Layer 0):**
```
output_0 = 0.5 * Triglyceride + 0.5 * VMA
```

**Pruned Model (Layer 0):**
```
output_0 = 0.0 * Triglyceride + 1.0 * VMA = VMA
```

### The Key Insight

Even with balanced weights (0.5/0.5), **the outputs are fundamentally different**:

- **Original output**: Average of Triglyceride and VMA
- **Pruned output**: Just VMA

**The difference is 0.5 × Triglyceride** - which is NOT negligible!

### Numerical Example

Using random test data (100 samples):

| Metric | Original (0.5×Trig + 0.5×VMA) | Pruned (1.0×VMA) | Difference |
|--------|-------------------------------|------------------|------------|
| Mean output | 0.4872 | 0.4983 | 0.0111 |
| Std deviation | 0.1634 | 0.2333 | Higher variance |
| Max difference | - | - | **0.3447** |
| Correlation | - | - | 0.6894 |

**Individual samples can differ by up to 0.34** (on 0-1 scale)!

### Why This Causes 5% Accuracy Drop

1. **The model learned with averaged values**: During training, all downstream layers learned to work with `0.5*Trig + 0.5*VMA` as their input

2. **Pruning changes the distribution**: Now downstream layers receive just `VMA`, which:
   - Has different mean and variance
   - Has different range of values
   - Is only 69% correlated with the original aggregated value

3. **Cascade effect**: This change propagates through ALL 37 subsequent layers, compounding the error

## Analogy

Imagine training a chef to cook with **half butter + half olive oil**. If you suddenly give them **only olive oil**, even though olive oil was "50% of the recipe", the dish will taste different because:
- The proportions changed (100% oil vs 50% oil + 50% butter)
- The flavor profile changed
- All subsequent cooking steps were calibrated for the mixed fat, not pure oil

## What "Balanced Weights" Actually Mean

**Common misconception**: "0.5/0.5 weights mean the feature doesn't matter"

**Reality**: 0.5/0.5 weights mean **BOTH features contribute equally**:
```
output = 0.5*Feature_A + 0.5*Feature_B
```

If you remove Feature_A:
```
output = 1.0*Feature_B  # Different from above!
```

The output changes by `0.5*Feature_A`, which can be significant.

## The Gallstone Model Case

Looking at your tree structure:

```
Layer 1: [Triglyceride]─0.50──[a=0.02272487]─0.50──[Visceral Muscle Area]
```

**Before pruning:**
- Aggregates Triglyceride and VMA with equal weight
- Output is their arithmetic mean (since a≈0.5)
- This averaged value feeds into 36 downstream layers

**After pruning:**
- Output is just VMA (no averaging)
- VMA values are different from averaged values
- All 36 downstream layers receive different inputs
- Accuracy drops from 78.12% → 73.35% (4.77% drop)

## Is This a Bug?

**NO!** This is correct behavior. The 5% accuracy drop represents the **true cost** of removing that feature.

The feature has:
- ✅ Low bias (close to disjunctive)
- ✅ Balanced weights (equal contribution)
- ❌ But still provides unique information worth 5% accuracy

## How to Interpret Pruning Results

When pruning shows a large accuracy drop even for "balanced" features, it means:

1. **The feature contains unique information** not captured by other features
2. **The aggregation itself matters**, not just the weights
3. **The model truly needs that feature** for optimal performance

### Actionable Insights

From your pruning analysis:
- Features 1-28: Can be pruned with minimal impact (73.35% → 73.04%)
- Features 29-30: Small impact (73.04% → 67.71%)
- Feature 31: Significant (67.71% → 67.08%)
- Features 32-36: Critical features

**Feature 0 (Triglyceride) is actually important** despite appearing "balanced"!

## Mathematical Proof

For balanced weights with arithmetic mean aggregator:

```
original(x₁, x₂) = 0.5x₁ + 0.5x₂
pruned(x₁, x₂) = 1.0x₂

difference = original - pruned
          = (0.5x₁ + 0.5x₂) - x₂
          = 0.5x₁ - 0.5x₂
          = 0.5(x₁ - x₂)
```

**The difference is proportional to (x₁ - x₂)**:
- If x₁ and x₂ are similar → small difference
- If x₁ and x₂ are different → large difference

In your case, Triglyceride and VMA are sufficiently different that the 5% accuracy drop is justified.

## Conclusion

✅ **The 5% accuracy drop is REAL and EXPECTED**
✅ **Pruning is working correctly**
✅ **Triglyceride is more important than it appears**
✅ **Balanced weights ≠ unimportant feature**

The feature appears to have "little weight" (0.5), but it's actually **sharing equal importance** with VMA. Removing it fundamentally changes what the model sees, causing the accuracy drop.
