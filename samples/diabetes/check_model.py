import torch
import sys
sys.path.insert(0, '../../')

from bacon.baconNet import baconNet
from bacon.transformationLayer import IdentityTransformation, NegationTransformation, PeakTransformation
from dataset import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data first to get feature count
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)

# Create transformations (must match what was used during training)
trans = [
    IdentityTransformation(1), 
    NegationTransformation(1), 
    PeakTransformation(1)
]

# Create model instance with matching configuration
print("Creating and loading model...")
model = baconNet(
    input_size=X_train.shape[1],
    aggregator="lsp.half_weight",
    use_transformation_layer=True,
    transformations=trans,
    weight_mode='fixed',
    weight_normalization='softmax'
)
model.load_model('assembler.pth')
model.eval()

# Test model
print("Testing model...")
with torch.no_grad():
    outputs = model(X_test)
    preds = (outputs > 0.5).float()
    acc = (preds == Y_test).float().mean().item()
    print(f'\nFull model accuracy: {acc * 100:.2f}%')
    
    # Also check individual outputs to compare
    print(f'\nFirst 10 predictions:')
    print(f'Model outputs: {outputs[:10].squeeze().cpu().numpy()}')
    print(f'Predictions:   {preds[:10].squeeze().cpu().numpy()}')
    print(f'Actual:        {Y_test[:10].squeeze().cpu().numpy()}')
