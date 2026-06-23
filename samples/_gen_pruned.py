"""Generate the critical (pruned) tree JSON + XML for a trained sample model.

Run from within a sample directory, e.g.::

    cd samples/heart-disease
    python ../_gen_pruned.py heart-disease

It loads the saved ``assembler.pth`` snapshot using the same configuration the
model was trained with, runs the pruning analysis to find the critical subtree,
and writes ``<sample>_pruned_tree.json`` / ``.xml`` next to the snapshot.
"""

import sys
import os

sys.path.insert(0, '../../')  # bacon package root
sys.path.insert(0, '../')      # common module
sys.path.insert(0, '.')        # local dataset module

import torch
import logging

from dataset import prepare_data
from common import create_bacon_model, analyze_feature_importance

logging.basicConfig(level=logging.INFO, format='%(message)s')

SAMPLE_CONFIGS = {
    'heart-disease': dict(
        title='Heart Disease',
        basename='heart_disease_pruned_tree',
        model_kwargs=dict(
            aggregator='lsp.half_weight',
            weight_mode='trainable',
            use_transformation_layer=True,
            weight_normalization='softmax',
            use_class_weighting=True,
        ),
    ),
    'breast-cancer': dict(
        title='Breast Cancer',
        basename='breast_cancer_pruned_tree',
        model_kwargs=dict(
            aggregator='lsp.full_weight',
            weight_mode='fixed',
            use_transformation_layer=False,
            weight_normalization='softmax',
            use_class_weighting=False,
            loss_amplifier=1000,
        ),
    ),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SAMPLE_CONFIGS:
        raise SystemExit(f"Usage: python ../_gen_pruned.py [{'|'.join(SAMPLE_CONFIGS)}]")

    key = sys.argv[1]
    cfg = SAMPLE_CONFIGS[key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)
    num_features = len(feature_names)
    print(f"\nLoaded {key}: {num_features} features, {X_test.shape[0]} test samples")

    model = create_bacon_model(input_size=num_features, **cfg['model_kwargs'])
    snapshot = os.path.abspath('assembler.pth')
    model.load_model(snapshot)
    model.eval()
    print(f"Loaded snapshot: {snapshot}")

    analyze_feature_importance(
        model,
        X_test, Y_test,
        feature_names,
        title_prefix=cfg['title'],
        threshold=0.5,
        baseline_enabled=True,
        device=device,
        save_pruned_tree=True,
        pruned_tree_basename=cfg['basename'],
    )

    print(f"\nGenerated {cfg['basename']}.json and {cfg['basename']}.xml")


if __name__ == '__main__':
    main()
