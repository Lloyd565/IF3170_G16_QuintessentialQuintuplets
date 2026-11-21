"""Gini impurity and impurity reduction for from-scratch CART classification."""

import numpy as np


def gini_impurity(class_counts):
    """Return Gini impurity from counts, optionally for a batch of nodes.

    The last axis contains counts for each class. An empty node has impurity 0.
    For example, ``gini_impurity([2, 2]) == 0.5``.
    """
    counts = np.asarray(class_counts, dtype=float)
    if counts.ndim == 0 or counts.shape[-1] == 0:
        raise ValueError("class_counts must have a nonempty class axis.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Class counts must be finite and nonnegative.")
    totals = counts.sum(axis=-1, keepdims=True)
    probabilities = np.divide(
        counts, totals, out=np.zeros_like(counts), where=totals != 0
    )
    return np.where(totals[..., 0] > 0, 1 - np.sum(probabilities**2, axis=-1), 0.0)


def evaluate_split(parent_counts, left_counts, right_counts):
    """Return parent Gini minus sample-weighted child Gini.

    Inputs contain class counts in the same class order, rather than labels.
    Children may have shape ``(n_candidates, n_classes)`` to score many splits.
    """
    parent = np.asarray(parent_counts, dtype=float)
    left = np.asarray(left_counts, dtype=float)
    right = np.asarray(right_counts, dtype=float)
    parent_impurity = gini_impurity(parent)
    left_impurity = gini_impurity(left)
    right_impurity = gini_impurity(right)
    if left.shape != right.shape or left.shape[-1] != parent.shape[-1]:
        raise ValueError("Parent and children must use the same class axis.")
    if not np.allclose(left + right, parent):
        raise ValueError("Child class counts must sum to the parent counts.")
    total = parent.sum(axis=-1)
    weighted = left.sum(axis=-1) * left_impurity + right.sum(axis=-1) * right_impurity
    return parent_impurity - np.divide(
        weighted, total, out=np.zeros_like(weighted), where=total != 0
    )
