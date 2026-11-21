"""Deterministic CART classifier implemented with NumPy and the standard library.

Numeric features use binary threshold splits; categorical features use binary
one-category-versus-rest splits. Missing-value routing is learned independently
at each split by minimizing weighted Gini impurity, using training data only.
"""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import pickle

import numpy as np

from .criteria import evaluate_split, gini_impurity


def _is_missing(value):
    """Recognize None, NaN, NaT, pandas.NA, and blank CSV fields without pandas."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return type(value).__name__ == "NAType"


@dataclass
class TreeNode:
    """A leaf's class distribution, or an internal node with two children."""

    class_counts: np.ndarray
    impurity: float
    n_samples: int
    feature_index: int | None = None
    split_kind: str | None = None
    split_value: float | None = None
    missing_go_left: bool = False
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self):
        return self.left is None


@dataclass
class _Split:
    feature_index: int
    kind: str
    value: float | None
    missing_go_left: bool
    gain: float


class DecisionTreeClassifier:
    """From-scratch CART with numeric, categorical, and missing features.

    ``fit(X, y)`` accepts a 2D array/list or DataFrame and 1D class labels.
    Numeric-looking strings are inferred as numeric; other columns are
    categorical. Override inference with a sequence of ``"numeric"`` and
    ``"categorical"`` in ``feature_types``, in the same order as X's columns.

    ``max_depth=None`` allows unlimited depth; 0 produces a single leaf.
    ``min_impurity_decrease`` uses impurity reduction weighted by the fraction
    of training samples reaching the node. Zero-gain splits are allowed at its
    default value of 0, so interactions such as XOR can still be learned.

    During prediction, unseen categories use the node's missing-value route.
    If no training values were missing at that node, the larger child is used
    (left on ties). Other ties use feature order, then candidate order.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        feature_types=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.feature_types = feature_types
        self.tree_ = None
        self._validate_parameters()

    def _validate_parameters(self):
        for name, minimum in (("max_depth", 0), ("min_samples_split", 2), ("min_samples_leaf", 1)):
            value = getattr(self, name)
            if name == "max_depth" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        try:
            decrease = float(self.min_impurity_decrease)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_impurity_decrease must be finite and nonnegative.") from exc
        if not np.isfinite(decrease) or decrease < 0:
            raise ValueError("min_impurity_decrease must be finite and nonnegative.")

    @staticmethod
    def _as_array(X):
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("X must be a 2D table with at least one feature.")
        return values

    def _encode_features(self, values, fitting):
        encoded = np.empty(values.shape, dtype=float)
        if fitting:
            types = self.feature_types
            if types is not None:
                if isinstance(types, str) or len(types) != values.shape[1]:
                    raise ValueError("feature_types must have one entry per feature.")
                if any(kind not in {"numeric", "categorical"} for kind in types):
                    raise ValueError("Feature types must be 'numeric' or 'categorical'.")
            self.feature_types_ = []
            self.categories_ = []
            self._category_maps = []

        for index in range(values.shape[1]):
            column = values[:, index]
            missing = np.fromiter((_is_missing(v) for v in column), dtype=bool, count=len(column))
            observed = column[~missing]
            if fitting:
                if types is None:
                    try:
                        np.asarray(observed, dtype=float)
                        kind = "numeric"
                    except (TypeError, ValueError):
                        kind = "categorical"
                else:
                    kind = types[index]
                self.feature_types_.append(kind)
                self.categories_.append([])
                self._category_maps.append({})
            kind = self.feature_types_[index]
            if kind == "numeric":
                try:
                    numbers = np.asarray(observed, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature {index} must contain numeric values or nulls.") from exc
                if not np.all(np.isfinite(numbers)):
                    raise ValueError(f"Feature {index} contains infinite numeric values.")
                encoded[missing, index] = np.nan
                encoded[~missing, index] = numbers
            else:
                mapping = self._category_maps[index]
                codes = np.full(len(column), -1.0)
                for row in np.flatnonzero(~missing):
                    value = column[row]
                    try:
                        if fitting and value not in mapping:
                            mapping[value] = len(mapping)
                            self.categories_[index].append(value)
                        codes[row] = mapping.get(value, -2)
                    except TypeError as exc:
                        raise ValueError(f"Feature {index} must contain scalar categories.") from exc
                encoded[:, index] = codes
        return encoded

    def fit(self, X, y):
        """Learn a tree and all feature metadata exclusively from X and y."""
        self._validate_parameters()
        self.tree_ = None
        values = self._as_array(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(values) or len(labels) == 0:
            raise ValueError("y must be 1D, nonempty, and have the same number of rows as X.")
        if any(_is_missing(value) for value in labels):
            raise ValueError("Class labels cannot contain missing values.")
        if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
            raise ValueError("Class labels must be finite.")
        self.classes_, targets = np.unique(labels, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = tuple(X.columns) if hasattr(X, "columns") else None
        encoded = self._encode_features(values, fitting=True)
        self._n_training_samples = len(values)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        self.n_nodes_ = 0
        self.n_leaves_ = 0
        self.tree_depth_ = 0
        self.tree_ = self._build_tree(encoded, targets, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _best_candidate(self, parent, left, right, missing):
        """Score candidate counts for both possible missing-value assignments."""
        if len(left) == 0:
            return None
        best = None
        directions = (False, True) if missing.sum() else (None,)
        for direction in directions:
            left_counts = left + missing if direction else left
            right_counts = right if direction else right + missing
            valid = (left_counts.sum(axis=1) >= self.min_samples_leaf) & (
                right_counts.sum(axis=1) >= self.min_samples_leaf
            )
            if not valid.any():
                continue
            gains = evaluate_split(parent, left_counts, right_counts)
            gains = np.where(valid, gains, -np.inf)
            index = int(np.argmax(gains))
            go_left = direction if direction is not None else left[index].sum() >= right[index].sum()
            candidate = (index, float(gains[index]), bool(go_left))
            if best is None or candidate[1] > best[1]:
                best = candidate
        return best

    def _find_split(self, X, y, parent):
        best = None
        for feature in range(self.n_features_in_):
            values = X[:, feature]
            numeric = self.feature_types_[feature] == "numeric"
            missing_mask = np.isnan(values) if numeric else values < 0
            observed = values[~missing_mask]
            labels = y[~missing_mask]
            observed_counts = np.bincount(labels, minlength=self.n_classes_)
            missing_counts = parent - observed_counts
            if len(observed) == 0:
                continue
            if numeric:
                order = np.argsort(observed, kind="stable")
                sorted_values = observed[order]
                positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
                cumulative = np.zeros((len(observed), self.n_classes_), dtype=np.int64)
                cumulative[np.arange(len(observed)), labels[order]] = 1
                np.cumsum(cumulative, axis=0, out=cumulative)
                left = cumulative[positions]
                low, high = sorted_values[positions], sorted_values[positions + 1]
                thresholds = low / 2 + high / 2
                split_values = np.where((thresholds >= high) | (thresholds < low), low, thresholds)
                kind = "numeric"
            else:
                split_values, inverse = np.unique(observed, return_inverse=True)
                left = np.zeros((len(split_values), self.n_classes_), dtype=np.int64)
                np.add.at(left, (inverse, labels), 1)
                kind = "categorical"
            right = observed_counts - left
            candidate = self._best_candidate(parent, left, right, missing_counts)
            if candidate is not None:
                index, gain, go_left = candidate
                if best is None or gain > best.gain:
                    best = _Split(feature, kind, float(split_values[index]), go_left, gain)

            # Missingness itself can be predictive even if all observed values
            # are identical, so also consider missing-versus-observed splits.
            if missing_counts.sum():
                candidate = self._best_candidate(
                    parent, np.zeros((1, self.n_classes_)), observed_counts[None, :], missing_counts
                )
                if candidate is not None and (best is None or candidate[1] > best.gain):
                    best = _Split(feature, "missing", None, True, candidate[1])
        return best

    def _left_mask(self, values, feature, kind, split_value, missing_go_left):
        missing = np.isnan(values) if self.feature_types_[feature] == "numeric" else values < 0
        if kind == "missing":
            return missing
        left = values <= split_value if kind == "numeric" else values == split_value
        return np.where(missing, missing_go_left, left)

    def _build_tree(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes_)
        node = TreeNode(counts, float(gini_impurity(counts)), len(y))
        self.n_nodes_ += 1
        self.tree_depth_ = max(self.tree_depth_, depth)
        stop = (
            np.count_nonzero(counts) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
        )
        split = None if stop else self._find_split(X, y, counts)
        if split is None or (
            len(y) / self._n_training_samples * split.gain + 1e-12
            < float(self.min_impurity_decrease)
        ):
            self.n_leaves_ += 1
            return node
        mask = self._left_mask(
            X[:, split.feature_index], split.feature_index, split.kind, split.value, split.missing_go_left
        )
        node.feature_index = split.feature_index
        node.split_kind = split.kind
        node.split_value = split.value
        node.missing_go_left = split.missing_go_left
        self.feature_importances_[split.feature_index] += len(y) * max(split.gain, 0)
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def _require_fitted(self):
        if self.tree_ is None:
            raise RuntimeError("Fit the decision tree before prediction or saving.")

    def predict_proba(self, X):
        """Return leaf class proportions, in the order of ``classes_``."""
        self._require_fitted()
        values = self._as_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features.")
        if (
            self.feature_names_in_ is not None
            and hasattr(X, "columns")
            and tuple(X.columns) != self.feature_names_in_
        ):
            raise ValueError("Prediction columns must match training column names and order.")
        encoded = self._encode_features(values, fitting=False)
        probabilities = np.empty((len(encoded), self.n_classes_), dtype=float)

        def visit(node, indices):
            if len(indices) == 0:
                return
            if node.is_leaf:
                probabilities[indices] = node.class_counts / node.n_samples
                return
            mask = self._left_mask(
                encoded[indices, node.feature_index], node.feature_index,
                node.split_kind, node.split_value, node.missing_go_left,
            )
            visit(node.left, indices[mask])
            visit(node.right, indices[~mask])

        visit(self.tree_, np.arange(len(encoded)))
        return probabilities

    def predict(self, X):
        """Return majority-class predictions; class ties follow ``classes_``."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X, y):
        """Return classification accuracy on a nonempty labelled dataset."""
        prediction = self.predict(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(prediction) or len(target) == 0:
            raise ValueError("y must be nonempty, 1D, and aligned with X.")
        return float(np.mean(prediction == target))

    def save(self, path):
        """Save the fitted tree, category mappings, and class order to a local file."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        """Load a trusted local pickle written by ``save``."""
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise ValueError("The file does not contain a DTL classifier.")
        model._require_fitted()
        return model


DTL = DecisionTreeClassifier


def dtl(X, y, **parameters):
    """Train and return a CART classifier through the package's original entry point."""
    return DecisionTreeClassifier(**parameters).fit(X, y)
