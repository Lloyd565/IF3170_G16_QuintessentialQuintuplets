import numpy as np


def entropy(class_counts):
    counts = np.asarray(class_counts, dtype=float)
    totals = counts.sum(axis=-1, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    logarithms = np.log2(np.where(probabilities > 0, probabilities, 1))
    return -np.sum(probabilities * logarithms, axis=-1)


def evaluate_split(parent_counts, left_counts, right_counts):
    parent = np.asarray(parent_counts, dtype=float)
    left = np.asarray(left_counts, dtype=float)
    right = np.asarray(right_counts, dtype=float)
    weighted = left.sum(axis=-1) * entropy(left) + right.sum(axis=-1) * entropy(right)
    remainder = np.divide(
        weighted, parent.sum(axis=-1), out=np.zeros_like(weighted), where=parent.sum(axis=-1) > 0,
    )
    return entropy(parent) - remainder
