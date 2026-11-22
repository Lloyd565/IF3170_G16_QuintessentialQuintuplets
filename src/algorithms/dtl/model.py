from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .criteria import evaluate_split


@dataclass
class TreeNode:
    class_counts: np.ndarray
    feature_index: int | None = None
    split_value: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class DecisionTreeClassifier:
    def __init__(
        self, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, min_impurity_decrease=0.0,
    ):
        for name, value, minimum in (
            ("max_depth", max_depth, 0),
            ("min_samples_split", min_samples_split, 2),
            ("min_samples_leaf", min_samples_leaf, 1),
        ):
            if name == "max_depth" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} harus berupa integer >= {minimum}.")
        if not np.isfinite(min_impurity_decrease) or min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease harus finite dan nonnegatif.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.tree_ = None

    @staticmethod
    def _features(X):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError("X harus berupa matriks numerik tanpa null; jalankan preprocessing.")
        return values

    def fit(self, X, y):
        self.tree_ = None
        values, target = self._features(X), np.asarray(y)
        if target.ndim != 1 or len(target) != len(values) or len(target) == 0:
            raise ValueError("y harus satu dimensi dan sesuai jumlah baris X.")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("Label is_canceled harus 0 atau 1.")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = values.shape[1]
        self.n_samples_ = len(values)
        self.tree_ = self._build_tree(values, target.astype(int), 0)
        return self

    def _best_split(self, X, y, counts):
        best = (None, None, -np.inf)
        for feature in range(self.n_features_in_):
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            positions = np.flatnonzero(values[:-1] < values[1:])
            sizes = positions + 1
            valid = (sizes >= self.min_samples_leaf) & (len(y) - sizes >= self.min_samples_leaf)
            positions, sizes = positions[valid], sizes[valid]
            if len(positions) == 0:
                continue
            positives = np.cumsum(y[order])[positions]
            left = np.column_stack((sizes - positives, positives))
            gains = evaluate_split(counts, left, counts - left)
            index = int(np.argmax(gains))
            if gains[index] > best[2]:
                position = positions[index]
                low, high = values[position], values[position + 1]
                threshold = low / 2 + high / 2
                if not low <= threshold < high:
                    threshold = low
                best = (feature, float(threshold), float(gains[index]))
        return best

    def _build_tree(self, X, y, depth):
        counts = np.bincount(y, minlength=2)
        node = TreeNode(counts)
        if (
            np.count_nonzero(counts) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < max(self.min_samples_split, 2 * self.min_samples_leaf)
        ):
            return node
        feature, threshold, gain = self._best_split(X, y, counts)
        if feature is None or len(y) / self.n_samples_ * gain + 1e-12 < self.min_impurity_decrease:
            return node
        left = X[:, feature] <= threshold
        node.feature_index, node.split_value = feature, threshold
        node.left = self._build_tree(X[left], y[left], depth + 1)
        node.right = self._build_tree(X[~left], y[~left], depth + 1)
        return node

    def predict_proba(self, X):
        if self.tree_ is None:
            raise RuntimeError("Latih model dengan fit() sebelum prediksi.")
        values = self._features(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Jumlah fitur prediksi harus sama dengan saat training.")
        probabilities = np.empty((len(values), 2))
        for index, row in enumerate(values):
            node = self.tree_
            while node.feature_index is not None:
                node = node.left if row[node.feature_index] <= node.split_value else node.right
            probabilities[index] = node.class_counts / node.class_counts.sum()
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


DTL = DecisionTreeClassifier


def dtl(X, y, **parameters):
    return DTL(**parameters).fit(X, y)
