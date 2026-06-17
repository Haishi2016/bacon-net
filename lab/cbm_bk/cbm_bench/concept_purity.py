"""Concept-purity / leakage metrics: OIS and NIS.

Faithful reimplementation of the two metrics from

    Espinosa Zarlenga et al., "Towards Robust Metrics for Concept
    Representation Evaluation", AAAI 2023 (arXiv:2301.10367),

as used by the IB-CBM evaluation in Galliamov et al., "Concepts'
Information Bottleneck Models", ICLR 2026 (arXiv:2602.14626).

Both metrics quantify *concept leakage* -- the extent to which a concept's
soft representation encodes information about *other* concepts. Lower is
better (0 == perfectly pure / no leakage).

* **Oracle Impurity Score (OIS)** -- builds a purity matrix whose ``(i, j)``
  entry is the test AUC of predicting ground-truth concept ``j`` from the soft
  representation of concept ``i`` alone, then measures the Frobenius distance
  to an *oracle* purity matrix built the same way from the ground-truth
  concept labels. Captures leakage encoded *within single* concepts.

* **Niche Impurity Score (NIS)** -- trains one predictor on all soft concepts
  and measures, across a sweep of correlation thresholds ``beta``, how well
  each concept can still be predicted from the concepts *outside* its
  correlation niche. Captures leakage distributed *across subsets* of
  concepts.

The reference implementation uses TensorFlow for the OIS predictor; this port
uses scikit-learn ``MLPClassifier`` for both metrics so the benchmark keeps a
single (already-present) dependency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings


def _default_predictor_fn(n_samples: int) -> Callable[[], MLPClassifier]:
    """Single-hidden-layer ReLU MLP, mirroring the reference OIS predictor."""

    def make() -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            max_iter=200,
            early_stopping=False,
            batch_size=min(512, max(8, n_samples)),
            random_state=1,
        )

    return make


def _safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC for a binary target, returning chance (0.5) on degenerate columns."""
    if len(np.unique(y_true)) < 2:
        # A constant target has no ROC; treat as uninformative.
        return 0.5
    return float(roc_auc_score(y_true, y_score))


@ignore_warnings(category=ConvergenceWarning)
def concept_purity_matrix(
    c_soft: np.ndarray,
    c_true: np.ndarray,
    predictor_fn: Callable[[], MLPClassifier] | None = None,
    test_size: float = 0.2,
    ignore_diags: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """Purity matrix ``P`` with ``P[i, j]`` = test AUC of predicting concept
    ``j`` from feature column ``i``.

    ``c_soft`` and ``c_true`` are ``(n_samples, n_concepts)`` arrays; ``c_soft``
    holds the soft representation used as the predictor input (for the oracle
    matrix this is the ground-truth labels themselves).
    """
    c_soft = np.asarray(c_soft, dtype=np.float32)
    c_true = np.asarray(c_true)
    n_samples, n_concepts = c_true.shape

    make_predictor = predictor_fn or _default_predictor_fn(n_samples)

    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=test_size, random_state=random_state
    )

    result = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    for i in range(n_concepts):
        x_train = c_soft[train_idx, i : i + 1]
        x_test = c_soft[test_idx, i : i + 1]
        for j in range(n_concepts):
            if ignore_diags and i == j:
                # Predicting a concept from its own ground truth is trivially
                # perfect; the oracle matrix fixes the diagonal to 1.
                result[i, j] = 1.0
                continue

            y_train = c_true[train_idx, j]
            y_test = c_true[test_idx, j]
            if len(np.unique(y_train)) < 2:
                # Cannot fit a classifier on a constant target.
                result[i, j] = 0.5
                continue

            clf = make_predictor()
            clf.fit(x_train, y_train)
            proba = clf.predict_proba(x_test)
            # Probability of the positive class.
            pos = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            result[i, j] = _safe_binary_auc(y_test, pos)

    return result


def oracle_impurity_score(
    c_soft: np.ndarray,
    c_true: np.ndarray,
    predictor_fn: Callable[[], MLPClassifier] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    return_matrices: bool = False,
):
    """Oracle Impurity Score (OIS).

    ``OIS = || oracle_matrix - purity_matrix ||_F / (n_concepts / 2)``.

    Returns a non-negative float (0 == pure). When ``return_matrices`` is True,
    returns ``(score, purity_matrix, oracle_matrix)``.
    """
    c_true = np.asarray(c_true)
    n_concepts = c_true.shape[1]

    purity = concept_purity_matrix(
        c_soft, c_true, predictor_fn, test_size, ignore_diags=False,
        random_state=random_state,
    )
    oracle = concept_purity_matrix(
        c_true, c_true, predictor_fn, test_size, ignore_diags=True,
        random_state=random_state,
    )
    impurity = np.linalg.norm(np.abs(oracle - purity), ord="fro")
    score = float(impurity / (n_concepts / 2.0))
    if return_matrices:
        return score, purity, oracle
    return score


def _niche_finding_corr(
    c_soft: np.ndarray, c_true: np.ndarray, threshold: float
) -> np.ndarray:
    """Correlation niches: ``niches[i, j]`` True iff ``|corr(soft_i, true_j)| >
    threshold``. Shape ``(n_concepts, n_concepts)``."""
    n_concepts = c_soft.shape[1]
    stacked = np.hstack([c_soft, c_true]).T
    corr = np.corrcoef(stacked)
    # Top-right block: rows = soft concepts, cols = true concepts.
    niching_matrix = corr[:n_concepts, n_concepts:]
    niching_matrix = np.nan_to_num(niching_matrix, nan=0.0)
    return np.abs(niching_matrix) > threshold


def _niche_impurity_at(
    c_soft_test: np.ndarray,
    c_true_test: np.ndarray,
    classifier: MLPClassifier,
    niches: np.ndarray,
) -> float:
    """Macro AUC of predicting each concept from concepts *outside* its niche."""
    n_concepts = c_true_test.shape[1]

    preds = []
    for j in range(n_concepts):
        outside = niches[:, j] <= 0  # concepts outside concept j's niche
        masked = np.zeros_like(c_soft_test)
        masked[:, outside] = c_soft_test[:, outside]
        proba_j = classifier.predict_proba(masked)
        preds.append(proba_j[:, j])

    y_preds = np.vstack(preds).T  # (n_samples, n_concepts)
    y_preds = softmax(y_preds, axis=1)

    # Macro AUC over concept columns with >= 2 classes (matches the reference
    # multilabel-indicator behaviour, robust to degenerate columns).
    aucs = []
    for j in range(n_concepts):
        if len(np.unique(c_true_test[:, j])) >= 2:
            aucs.append(roc_auc_score(c_true_test[:, j], y_preds[:, j]))
    return float(np.mean(aucs)) if aucs else 0.5


@ignore_warnings(category=ConvergenceWarning)
def niche_impurity_score(
    c_soft: np.ndarray,
    c_true: np.ndarray,
    predictor_fn: Callable[[], MLPClassifier] | None = None,
    delta_beta: float = 0.05,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """Niche Impurity Score (NIS).

    Trains one predictor on all soft concepts, then integrates (trapezoidal)
    the out-of-niche predictability over correlation thresholds
    ``beta in [0, 1)``. Lower is better (less cross-niche leakage).
    """
    c_soft = np.asarray(c_soft, dtype=np.float32)
    c_true = np.asarray(c_true)
    n_samples, n_concepts = c_true.shape

    def make_predictor() -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(20, 20),
            max_iter=1000,
            batch_size=min(512, max(8, n_samples)),
            random_state=1,
        )

    make = predictor_fn or make_predictor

    c_soft_train, c_soft_test, c_true_train, c_true_test = train_test_split(
        c_soft, c_true, test_size=test_size, random_state=random_state
    )

    classifier = make()
    classifier.fit(c_soft_train, c_true_train)

    auc = 0.0
    prev_value = None
    for beta in np.arange(0.0, 1.0, delta_beta):
        niches = _niche_finding_corr(c_soft_train, c_true_train, float(beta))
        value = _niche_impurity_at(c_soft_test, c_true_test, classifier, niches)
        if prev_value is not None:
            auc += (prev_value + value) * (delta_beta / 2.0)
        prev_value = value

    return float(auc)
