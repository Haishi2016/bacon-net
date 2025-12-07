"""
Example: Using Peak Transformation for Age-Sensitive Credit Risk

This demonstrates using the peak transformation for features where
an optimal mid-range value exists. For example:
- Age: Too young (risky, inexperienced) or too old (fixed income) may be less favorable
- Income: Very low (can't repay) or very high (may not need loan) could be less ideal  
- Credit utilization: 0% (no history) or 100% (maxed out) both problematic

The peak transformation learns the optimal value for each feature.

This example uses a simple neural network with TransformationLayer directly,
without the full BaconNet framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bacon.transformationLayer import (
    TransformationLayer,
    IdentityTransformation,
    NegationTransformation,
    PeakTransformation
)


class SimpleClassifier(nn.Module):
    """Simple neural network with transformation layer."""
    
    def __init__(self, num_features, num_classes, transformations=None):
        super().__init__()
        
        # Transformation layer
        if transformations is None:
            transformations = [
                IdentityTransformation(num_features),
                NegationTransformation(num_features),
                PeakTransformation(num_features)
            ]
        
        self.transform = TransformationLayer(
            num_features=num_features,
            transformations=transformations,
            temperature=1.0,
            device='cpu'
        )
        
        # Simple MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        x_transformed = self.transform(x)
        return self.classifier(x_transformed)


def create_synthetic_credit_data(n_samples=1000):
    """
    Create synthetic credit approval data with non-linear age relationship.
    
    Age around 35-45 (normalized: 0.35-0.45) has highest approval rate.
    Too young (<25) or too old (>60) have lower approval rates.
    """
    torch.manual_seed(42)
    
    # Features: [age_norm, income_norm, credit_score_norm]
    # Age: normalized from 18-80 to [0, 1]
    age_raw = torch.rand(n_samples) * 62 + 18  # 18-80 years
    age_norm = (age_raw - 18) / 62
    
    # Income: normalized log-scale
    income_raw = torch.exp(torch.rand(n_samples) * 5 + 8)  # $3k - $150k
    income_norm = (torch.log(income_raw) - 8) / 5
    
    # Credit score: 300-850
    credit_raw = torch.rand(n_samples) * 550 + 300
    credit_norm = (credit_raw - 300) / 550
    
    X = torch.stack([age_norm, income_norm, credit_norm], dim=1)
    
    # Target: approval probability based on features
    # Age has peak around 0.4 (about 37 years old)
    age_score = 1.0 - torch.abs(age_norm - 0.4)
    
    # Income: monotonic (higher is better, but saturates)
    income_score = torch.sigmoid((income_norm - 0.3) * 5)
    
    # Credit: monotonic (higher is better)
    credit_score = credit_norm
    
    # Combined score
    combined_score = (age_score * 0.3 + income_score * 0.3 + credit_score * 0.4)
    
    # Binary approval (with some noise)
    approval_prob = torch.sigmoid((combined_score - 0.6) * 10)
    y = (torch.rand(n_samples) < approval_prob).long()
    
    return X, y, age_raw, income_raw, credit_raw


def main():
    print("=" * 70)
    print("PEAK TRANSFORMATION FOR CREDIT RISK")
    print("=" * 70)
    print()
    
    # Generate data
    print("Generating synthetic credit data...")
    X_train, y_train, age_raw, income_raw, credit_raw = create_synthetic_credit_data(1000)
    X_test, y_test, _, _, _ = create_synthetic_credit_data(200)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: [age_norm, income_norm, credit_score_norm]")
    print()
    
    # Feature statistics
    print("Feature ranges (training data):")
    print(f"  Age: {age_raw.min():.1f} - {age_raw.max():.1f} years")
    print(f"  Income: ${income_raw.min():.0f} - ${income_raw.max():.0f}")
    print(f"  Credit Score: {credit_raw.min():.0f} - {credit_raw.max():.0f}")
    print()
    
    # Create model with peak transformation for age feature
    print("Creating model with peak transformation...")
    print("  - Identity transformation: f(x) = x")
    print("  - Negation transformation: f(x) = 1-x")
    print("  - Peak transformation: f(x) = 1 - |x - t| where t is learned")
    print()
    
    model = SimpleClassifier(num_features=3, num_classes=2)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Training
    print("Training for 200 epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0:
            # Evaluate
            model.eval()
            with torch.no_grad():
                train_pred = (outputs.argmax(dim=1) == y_train).float().mean()
                test_outputs = model(X_test)
                test_pred = (test_outputs.argmax(dim=1) == y_test).float().mean()
            
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, "
                  f"train_acc={train_pred:.3f}, test_acc={test_pred:.3f}")
    
    print()
    
    # Analyze learned transformations
    print("=" * 70)
    print("LEARNED TRANSFORMATIONS")
    print("=" * 70)
    print()
    
    summary = model.transform.get_transformation_summary()
    
    feature_names = ['Age (norm)', 'Income (norm)', 'Credit Score (norm)']
    
    for feat_idx in range(3):
        info = summary[feat_idx]
        print(f"{feature_names[feat_idx]}:")
        print(f"  Selected: {info['transformation']} (prob={info['probability']:.3f})")
        
        # Show all transformation probabilities
        transform_names = ['identity', 'negation', 'peak']
        for t_idx, (name, prob) in enumerate(zip(transform_names, info['all_probs'])):
            print(f"    {name:10s}: {prob:.3f}")
        
        # Show parameters if any
        if info['params']:
            for param_name, param_value in info['params'].items():
                print(f"  Learned {param_name}: {param_value}")
                
                # Interpret peak location for age
                if feat_idx == 0 and 'peak' in param_name:
                    # Extract numeric value from "peak_location=0.xxx" string
                    value_str = param_value.split('=')[1]
                    peak_norm = float(value_str)
                    age_years = 18 + peak_norm * 62
                    print(f"    → Optimal age: ~{age_years:.1f} years")
        
        print()
    
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("The model should learn:")
    print("  - Age: PEAK transformation (optimal around 35-45 years)")
    print("  - Income: IDENTITY transformation (higher is better)")
    print("  - Credit Score: IDENTITY transformation (higher is better)")
    print()
    
    # Final test accuracy
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_accuracy = (test_outputs.argmax(dim=1) == y_test).float().mean()
    
    print(f"Final test accuracy: {test_accuracy:.3f}")
    print()


if __name__ == '__main__':
    main()
