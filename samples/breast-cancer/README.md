

##  Saved models
| Model file | Accuracy | freeze_loss_threshold| weight_model| weight_penalty_strength| aggregator | weight_normalization|
|--------|--------|--------|--------|--------|--------|--------|
| assembler.pth-9297-minmax-e4 | 92.97% | 0.42 | trainable | 1e-4 | bool.min_max | softmax |
| assembler.pth-9350-minmax-e4 | 93.15% | 0.42 | trainable | 1e-4 | bool.min_max | softmax | 
| assembler.pth-9350-minmax-e4 | 93.50% | 0.42 | trainable | 1e-4 | bool.min_max | minmax | 
| assembler.pth-9402-minmax-fixed| 94.02% | 0.42 | fixed | - | bool.min_max | softmax |
| assembler.pth-9596-softmax-e4 | 95.96% | 0.035 | trainable | 1e-4 | lsp.half_weight | softmax |
| assembler.pth-9666-softmax-e4 | 96.66% | 0.052 | trainable | 1e-4 | lsp.half_weight | softmax |
| assembler.pth-9684-softmax-e4 | 96.84% | 0.07 | trainable | 1e-4 | lsp.full_weight | softmax |
| assembler.pth-9736-softmax-e4 | 97.36% | 0.05 | trainable | 1e-4 | lsp.full_weight | softmax |
| assembler.pth-9754-softmax-e2 | 97.54% | 0.07 | trainable | 1e-2 | lsp.half_weight | softmax |
| assembler.pth-9789-softmax-e4 | 97.89% | 0.05 | trainable | 1e-4 | lsp.full_weight | softmax |
| assembler.pth-9789-softmax-e4-2 | 97.89% | 0.05 | trainable | 1e-4 | lsp.full_weight | softmax |
| assembler.pth-9807-softmax-e4 | 98.07% | 0.05 | trainable | 1e-4 | lsp.full_weight | softmax |
| assembler.pth-9807-softmax-e4-2 | 98.07% | 0.04 | trainable | 1e-4 | lsp.full_weight | softmax |
| assembler.pth-9824-softmax-e4 | 98.24% | 0.05 | trainable | 1e-4 | lsp.full_weight | softmax |






